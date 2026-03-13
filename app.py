"""
FACE RECOGNITION SYSTEM - Flask Web Application
Provides REST API + Web UI for managing face recognition with Supabase backend.
"""

import os
import sys
import json
import base64
import time
import numpy as np
import cv2
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS

# Ensure project modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from detector import FaceDetector
from recognizer import FaceRecognizer
from supabase_db import SupabaseDatabase

# ==================== CONFIG ====================
# Supabase credentials - CHANGE THESE!
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://fpcrjmsekkhvjmpchfsf.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZwY3JqbXNla2todmptcGNoZnNmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzM0MjQ2NjAsImV4cCI6MjA4OTAwMDY2MH0.zMrGzyluVEeUT-veZEqoANj4RCKoUQQaSzRbpDxrCRs")

# ==================== APP INIT ====================
app = Flask(__name__, 
            template_folder="templates",
            static_folder="static")
CORS(app)

# Global objects (lazy init)
_detector = None
_recognizer = None
_db = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = FaceDetector(mode="image", num_faces=1)
    return _detector


def get_recognizer():
    global _recognizer
    if _recognizer is None:
        _recognizer = FaceRecognizer()
    return _recognizer


def get_db():
    global _db
    if _db is None:
        _db = SupabaseDatabase(SUPABASE_URL, SUPABASE_KEY)
        _db.sync_from_supabase()
    return _db


# ==================== API: POSE CHECK (for enrollment) ====================

@app.route("/api/check_pose", methods=["POST"])
def api_check_pose():
    """Quick head pose validation from a single frame.
    Used by enrollment UI to gate captures by direction.
    Reuses FaceInfo.check_pose() from detector.py (same as main.py).
    """
    try:
        data = request.get_json()
        b64_str = data.get("image", "")
        direction = data.get("direction", "straight")

        if "," in b64_str:
            b64_str = b64_str.split(",")[1]

        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"valid": False, "reason": "invalid_image"})

        detector = get_detector()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect(rgb)

        if len(faces) != 1:
            return jsonify({"valid": False, "reason": "no_face" if len(faces) == 0 else "multiple_faces"})

        face = faces[0]
        h_off, v_ratio = face.head_pose()
        pose_ok, pose_msg = face.check_pose(direction)

        return jsonify({
            "valid": bool(pose_ok),
            "h_offset": round(float(h_off), 3),
            "v_ratio": round(float(v_ratio), 3),
            "direction": direction,
            "message": pose_msg
        })

    except Exception as e:
        return jsonify({"valid": False, "reason": str(e)})


# ==================== WEB PAGES ====================

@app.route("/")
def index():
    return render_template("index.html")


# ==================== API: USERS ====================

@app.route("/api/users", methods=["GET"])
def api_get_users():
    """Get all registered users."""
    try:
        db = get_db()
        users = db.get_users()
        return jsonify({
            "success": True,
            "users": [{"name": n, "embeddings": c} for n, c in users.items()],
            "total_vectors": db.total
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/users/<name>", methods=["DELETE"])
def api_delete_user(name):
    """Delete a user and all their embeddings."""
    try:
        db = get_db()
        db.remove_user(name)
        return jsonify({"success": True, "message": f"User '{name}' deleted"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: ENROLL ====================

@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    """Enroll a user from uploaded images.
    
    Expects multipart form with:
    - name: user name
    - images: multiple image files
    """
    try:
        name = request.form.get("name", "").strip()
        if not name:
            return jsonify({"success": False, "error": "Name is required"}), 400

        images = request.files.getlist("images")
        if not images or len(images) < 3:
            return jsonify({"success": False, "error": "At least 3 images required"}), 400

        detector = get_detector()
        recognizer = get_recognizer()
        db = get_db()

        embeddings = []
        scores = []
        processed = 0
        skipped = 0

        for img_file in images:
            # Read image
            img_bytes = img_file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                skipped += 1
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect(rgb)

            if len(faces) != 1:
                skipped += 1
                continue

            face = faces[0]
            q_score, ok, reason = face.quality_check(img)

            if not ok:
                skipped += 1
                continue

            emb = recognizer.get_embedding(img, face.lm5)
            if emb is not None:
                embeddings.append(emb)
                scores.append(q_score)
                processed += 1

        if len(embeddings) < 3:
            return jsonify({
                "success": False,
                "error": f"Only {len(embeddings)} valid faces found. Need at least 3.",
                "processed": processed,
                "skipped": skipped
            }), 400

        # Clean outliers
        cleaned = recognizer.clean_embeddings(embeddings, OUTLIER_STD)
        db.add_user(name, cleaned, scores[:len(cleaned)])

        return jsonify({
            "success": True,
            "message": f"User '{name}' enrolled successfully",
            "total_embeddings": len(cleaned),
            "processed": processed,
            "skipped": skipped
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: ENROLL FROM BASE64 ====================

@app.route("/api/enroll/base64", methods=["POST"])
def api_enroll_base64():
    """Enroll a user from base64-encoded images (webcam captures).
    Uses PARALLEL image decoding + detection for speed.
    
    Expects JSON:
    {
        "name": "user_name",
        "images": ["base64_data_1", "base64_data_2", ...]
    }
    """
    try:
        import concurrent.futures

        t_start = time.time()
        data = request.get_json()
        name = data.get("name", "").strip()
        images_b64 = data.get("images", [])

        if not name:
            return jsonify({"success": False, "error": "Name is required"}), 400

        if len(images_b64) < 3:
            return jsonify({"success": False, "error": "At least 3 images required"}), 400

        detector = get_detector()
        recognizer = get_recognizer()
        db = get_db()

        # Step 1: PARALLEL image decoding (I/O bound — benefits from threading)
        def decode_image(b64_str):
            """Decode base64 to cv2 image."""
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            img_bytes = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        t1 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            decoded_images = list(pool.map(decode_image, images_b64))
        print(f"  [PERF] Decoded {len(decoded_images)} images in {time.time()-t1:.2f}s")

        # Step 2: Sequential detection + embedding (model is not thread-safe)
        t2 = time.time()
        embeddings = []
        scores = []
        processed = 0
        skipped = 0

        for img in decoded_images:
            if img is None:
                skipped += 1
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect(rgb)

            if len(faces) != 1:
                skipped += 1
                continue

            face = faces[0]
            q_score, ok, reason = face.quality_check(img)

            if not ok:
                skipped += 1
                continue

            emb = recognizer.get_embedding(img, face.lm5)
            if emb is not None:
                embeddings.append(emb)
                scores.append(q_score)
                processed += 1

        print(f"  [PERF] Processed {processed} faces in {time.time()-t2:.2f}s (skipped: {skipped})")

        if len(embeddings) < 3:
            return jsonify({
                "success": False,
                "error": f"Only {len(embeddings)} valid faces found. Need at least 3.",
                "processed": processed,
                "skipped": skipped
            }), 400

        # Step 3: Select top-K best embeddings by quality + remove outliers
        t3 = time.time()
        from config import ENROLL_KEEP_TOP
        final_embs, final_scores = recognizer.select_best_embeddings(
            embeddings, scores, keep_top=ENROLL_KEEP_TOP
        )
        db.add_user(name, final_embs, final_scores)
        print(f"  [PERF] Upload to Supabase in {time.time()-t3:.2f}s")
        print(f"  [ENROLL] Pipeline: {len(images_b64)} captured → {processed} valid → {len(final_embs)} saved (top-{ENROLL_KEEP_TOP})")

        total_time = time.time() - t_start
        print(f"  [PERF] Total enrollment: {total_time:.2f}s")

        return jsonify({
            "success": True,
            "message": f"User '{name}' enrolled successfully",
            "total_embeddings": len(final_embs),
            "captured": len(images_b64),
            "processed": processed,
            "skipped": skipped,
            "quality_range": f"{min(final_scores):.3f} ~ {max(final_scores):.3f}" if final_scores else "N/A",
            "time_seconds": round(total_time, 2)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: RECOGNIZE (single frame - backward compat) ====================

@app.route("/api/recognize", methods=["POST"])
def api_recognize_single():
    """Recognize from a single image (backward compatible)."""
    try:
        detector = get_detector()
        recognizer = get_recognizer()
        db = get_db()
        
        img = None

        if "image" in request.files:
            img_file = request.files["image"]
            img_bytes = img_file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            data = request.get_json(silent=True)
            if data and "image" in data:
                b64_str = data["image"]
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                img_bytes = base64.b64decode(b64_str)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"success": False, "error": "No valid image provided"}), 400

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect(rgb)

        results = []
        for face in faces:
            q_score, ok, reason = face.quality_check(img)
            if not ok:
                results.append({
                    "name": "Unknown",
                    "score": 0,
                    "quality": reason,
                    "bbox": [int(x) for x in face.bbox]
                })
                continue

            emb = recognizer.get_embedding(img, face.lm5)
            if emb is not None:
                name, score = db.match(emb)
                score = float(score)
                accepted = bool(score >= THRESHOLD)

                result = {
                    "name": name if accepted else "Unknown",
                    "score": round(score, 4),
                    "accepted": accepted,
                    "quality_score": round(float(q_score), 4),
                    "bbox": [int(x) for x in face.bbox]
                }
                results.append(result)

                if accepted:
                    db.log_attendance(name, score)

        return jsonify({
            "success": True,
            "faces_detected": len(faces),
            "results": results
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: RECOGNIZE MULTI-FRAME (production) ====================

@app.route("/api/recognize/multi", methods=["POST"])
def api_recognize_multi():
    """Multi-frame voting recognition for production accuracy.
    
    Pipeline:
      N frames → detect face each → embedding each → 
      average similarity per person → final decision
    
    Expects JSON:
    {
        "images": ["base64_1", "base64_2", ...],  // 5-10 frames
        "threshold": 0.45  // optional override
    }
    """
    try:
        import concurrent.futures

        t_start = time.time()
        data = request.get_json()
        images_b64 = data.get("images", [])
        threshold = data.get("threshold", THRESHOLD)

        if len(images_b64) < 2:
            return jsonify({"success": False, "error": "Need at least 2 frames for multi-frame recognition"}), 400

        detector = get_detector()
        recognizer = get_recognizer()
        db = get_db()

        # Step 1: Parallel decode
        def decode_image(b64_str):
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            img_bytes = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            decoded_images = list(pool.map(decode_image, images_b64))

        # Step 2: Extract embeddings from each frame
        frame_results = []  # per-frame: { name, score, emb, quality }
        valid_embeddings = []
        valid_scores = []

        for i, img in enumerate(decoded_images):
            if img is None:
                frame_results.append({"frame": i, "status": "decode_error"})
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect(rgb)

            if len(faces) != 1:
                frame_results.append({"frame": i, "status": "no_single_face", "faces": len(faces)})
                continue

            face = faces[0]
            q_score, q_ok, q_reason = face.quality_check(img)

            emb = recognizer.get_embedding(img, face.lm5)
            if emb is None:
                frame_results.append({"frame": i, "status": "embedding_failed"})
                continue

            # L2 normalize (already done in get_embedding, but be explicit)
            emb = emb / np.linalg.norm(emb)

            # Match against DB
            name, score = db.match(emb)
            score = float(score)

            valid_embeddings.append(emb)
            valid_scores.append(score)

            frame_results.append({
                "frame": i,
                "status": "ok",
                "name": name,
                "score": round(score, 4),
                "quality": round(float(q_score), 4) if q_ok else q_reason
            })

        if len(valid_scores) == 0:
            return jsonify({
                "success": True,
                "recognized": False,
                "name": "Unknown",
                "avg_score": 0,
                "reason": "No valid faces detected in any frame",
                "frames_total": len(images_b64),
                "frames_valid": 0,
                "frame_details": frame_results
            })

        # Step 3: Multi-frame voting
        # Average the top scores (remove worst outliers)
        sorted_scores = sorted(valid_scores, reverse=True)
        # Use top 70% of scores (remove worst 30%)
        keep_count = max(2, int(len(sorted_scores) * 0.7))
        top_scores = sorted_scores[:keep_count]
        avg_score = float(np.mean(top_scores))

        # Get consensus name (majority voting)
        name_votes = {}
        for fr in frame_results:
            if fr.get("status") == "ok" and fr.get("name"):
                n = fr["name"]
                name_votes[n] = name_votes.get(n, 0) + 1

        best_name = max(name_votes, key=name_votes.get) if name_votes else "Unknown"
        accepted = avg_score >= threshold

        # Step 4: Average embedding for even higher accuracy
        avg_emb = np.mean(valid_embeddings, axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        avg_name, avg_emb_score = db.match(avg_emb)
        avg_emb_score = float(avg_emb_score)

        # Use the HIGHER of the two methods
        final_score = max(avg_score, avg_emb_score)
        final_name = avg_name if avg_emb_score > avg_score else best_name
        final_accepted = final_score >= threshold

        elapsed = time.time() - t_start

        # Log attendance if recognized
        if final_accepted:
            db.log_attendance(final_name, final_score)

        print(f"  [RECOG] Multi-frame: {len(valid_scores)}/{len(images_b64)} frames valid")
        print(f"  [RECOG] Score voting: {avg_score:.4f} | Avg emb: {avg_emb_score:.4f} | Final: {final_score:.4f}")
        print(f"  [RECOG] Result: {final_name} ({'ACCEPTED' if final_accepted else 'REJECTED'}) in {elapsed:.2f}s")

        return jsonify({
            "success": True,
            "recognized": final_accepted,
            "name": final_name if final_accepted else "Unknown",
            "score": round(final_score, 4),
            "avg_score_voting": round(avg_score, 4),
            "avg_score_embedding": round(avg_emb_score, 4),
            "accepted": final_accepted,
            "frames_total": len(images_b64),
            "frames_valid": len(valid_scores),
            "per_frame_scores": [round(s, 4) for s in valid_scores],
            "frame_details": frame_results,
            "time_seconds": round(elapsed, 2),
            "method": "multi_frame_voting"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: ATTENDANCE ====================

@app.route("/api/attendance", methods=["GET"])
def api_attendance():
    """Get attendance logs."""
    try:
        db = get_db()
        limit = request.args.get("limit", 50, type=int)
        logs = db.get_attendance_logs(limit=limit)
        return jsonify({
            "success": True,
            "logs": logs
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: SYNC ====================

@app.route("/api/sync", methods=["POST"])
def api_sync():
    """Sync FAISS index from Supabase."""
    try:
        db = get_db()
        db.sync_from_supabase()
        return jsonify({
            "success": True,
            "message": "FAISS index synced from Supabase",
            "total_vectors": db.total
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: SYSTEM INFO ====================

@app.route("/api/info", methods=["GET"])
def api_info():
    """Get system information."""
    try:
        db = get_db()
        recognizer = get_recognizer()
        users = db.get_users()
        return jsonify({
            "success": True,
            "system": {
                "version": "4.0",
                "device": recognizer.device,
                "tta_enabled": TTA_ENABLED,
                "threshold": THRESHOLD,
                "total_users": len(users),
                "total_vectors": db.total,
                "database": "Supabase"
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("   FACE RECOGNITION SYSTEM - Web Server")
    print("   Flask + Supabase + MediaPipe + ArcFace")
    print("=" * 56)
    
    if SUPABASE_URL == "YOUR_SUPABASE_URL":
        print("\n  [WARNING] Supabase URL not configured!")
        print("  Set SUPABASE_URL and SUPABASE_KEY in app.py or as env vars")
        print("  Example:")
        print("    set SUPABASE_URL=https://xxxxx.supabase.co")
        print("    set SUPABASE_KEY=eyJhbGci...")
        print()

    app.run(host="0.0.0.0", port=5000, debug=True)
