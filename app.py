"""
FACE RECOGNITION SYSTEM v5.3 - Flask + SocketIO Web Application
Provides REST API + SocketIO Realtime + Web UI for face recognition.
Enhanced: SocketIO realtime recognition, Service Layer facade, Prometheus metrics.
"""

import os
import sys
import json
import base64
import time
import threading
import numpy as np
import cv2
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Ensure project modules are importable
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, 'core'))

# Load .env file (Supabase credentials, etc.)
from dotenv import load_dotenv
load_dotenv(os.path.join(base_dir, '.env'))

from config import *
from logger import get_logger
from detector import FaceDetector
from recognizer import FaceRecognizer
from metrics import metrics, metrics_endpoint

logger = get_logger("app")

# ==================== CONFIG ====================
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# ==================== APP INIT ====================
app = Flask(__name__, 
            template_folder="templates",
            static_folder="static")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    ping_timeout=30, ping_interval=10,
                    max_http_buffer_size=5 * 1024 * 1024)  # 5MB max frame

# Global service (lazy init)
_service = None


def get_service():
    """Get or create the FaceService singleton."""
    global _service
    if _service is None:
        from service import FaceService
        
        detector = FaceDetector(mode="image", num_faces=1)
        recognizer = FaceRecognizer()
        
        # Select database backend based on config
        if DB_BACKEND == "pgvector":
            from pgvector_db import PgVectorDatabase
            db = PgVectorDatabase(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Using pgvector cloud-native backend")
        else:
            from supabase_db import SupabaseDatabase
            db = SupabaseDatabase(SUPABASE_URL, SUPABASE_KEY)
            db.sync_from_supabase()
            logger.info("Using FAISS local backend")
        
        from anti_spoof import AntiSpoofer
        
        # Load Anti-Spoofing model (dùng path từ config)
        spoofer = None
        if os.path.exists(ANTI_SPOOF_PATH):
            spoofer = AntiSpoofer(ANTI_SPOOF_PATH, threshold=ANTI_SPOOF_THRESHOLD)
            logger.info(f"AntiSpoof loaded: {os.path.basename(ANTI_SPOOF_PATH)} (threshold={ANTI_SPOOF_THRESHOLD})")
            
        _service = FaceService(detector, recognizer, db, spoofer=spoofer)
        
        # Update Prometheus gauges
        users = _service.get_users()
        metrics.set_active_users(len(users))
    
    return _service


# ==================== API: POSE CHECK (for enrollment) ====================

@app.route("/api/check_pose", methods=["POST"])
def api_check_pose():
    """Quick head pose validation from a single frame."""
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

        svc = get_service()
        result = svc.check_pose(img, direction)
        return jsonify(result)

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
        svc = get_service()
        users = svc.get_users()
        return jsonify({
            "success": True,
            "users": [{"name": n, "embeddings": c} for n, c in users.items()]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/users/<name>", methods=["DELETE"])
def api_delete_user(name):
    """Delete a user and all their embeddings."""
    try:
        svc = get_service()
        svc.delete_user(name)
        return jsonify({"success": True, "message": f"User '{name}' deleted"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: ENROLL ====================

@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    """Enroll a user from uploaded images."""
    try:
        name = request.form.get("name", "").strip()
        if not name:
            return jsonify({"success": False, "error": "Name is required"}), 400

        images = request.files.getlist("images")
        if not images or len(images) < 3:
            return jsonify({"success": False, "error": "At least 3 images required"}), 400

        images_bgr = []
        for img_file in images:
            img_bytes = img_file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            images_bgr.append(img)

        svc = get_service()
        result = svc.enroll_user(name, images_bgr)

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: ENROLL FROM BASE64 ====================

@app.route("/api/enroll/base64", methods=["POST"])
def api_enroll_base64():
    """Enroll a user from base64-encoded images."""
    try:
        import concurrent.futures

        data = request.get_json()
        name = data.get("name", "").strip()
        images_b64 = data.get("images", [])

        if not name:
            return jsonify({"success": False, "error": "Name is required"}), 400

        if len(images_b64) < 3:
            return jsonify({"success": False, "error": "At least 3 images required"}), 400

        def decode_image(b64_str):
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            img_bytes = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            decoded_images = list(pool.map(decode_image, images_b64))

        svc = get_service()
        result = svc.enroll_user(name, decoded_images)
        result["captured"] = len(images_b64)

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: RECOGNIZE (single frame - backward compat) ====================

@app.route("/api/recognize", methods=["POST"])
def api_recognize_single():
    """Recognize from a single image."""
    try:
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

        svc = get_service()
        result = svc.recognize_single(img)

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: RECOGNIZE MULTI-FRAME (production) ====================

@app.route("/api/recognize/multi", methods=["POST"])
def api_recognize_multi():
    """Multi-frame voting recognition."""
    try:
        data = request.get_json(silent=True) or {}
        b64_images = data.get("images", [])
        threshold = data.get("threshold", THRESHOLD)

        if not b64_images:
            return jsonify({"success": False, "error": "No images provided"}), 400

        images_bgr = []
        for b64 in b64_images:
            try:
                if "," in b64:
                    b64 = b64.split(",")[1]
                img_bytes = base64.b64decode(b64)
                arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                images_bgr.append(img)
            except:
                images_bgr.append(None)

        svc = get_service()
        result = svc.recognize_multi(images_bgr, threshold=threshold)

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: ATTENDANCE ====================

@app.route("/api/attendance", methods=["GET"])
def api_attendance():
    """Get attendance logs."""
    try:
        svc = get_service()
        limit = request.args.get("limit", 50, type=int)
        logs = svc.get_attendance_logs(limit=limit)
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
        svc = get_service()
        svc.sync_db()
        return jsonify({"success": True, "message": "Database synced"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== API: SYSTEM INFO ====================

@app.route("/api/info", methods=["GET"])
def api_info():
    """Get system information."""
    try:
        svc = get_service()
        info = svc.get_system_info()
        return jsonify({"success": True, "system": info})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== PROMETHEUS METRICS ====================

@app.route("/metrics")
def prometheus_metrics():
    """Prometheus scraping endpoint."""
    return metrics_endpoint()


# ==================== SOCKETIO: REALTIME RECOGNITION ====================

# Lock to prevent concurrent recognition on the same connection
_recog_lock = threading.Lock()


@socketio.on('connect')
def handle_connect():
    logger.info(f"SocketIO client connected: {request.sid}")
    emit('server_ready', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"SocketIO client disconnected: {request.sid}")


@socketio.on('recognize_frame')
def handle_recognize_frame(data):
    """Realtime recognition: client sends 1 frame, server responds immediately."""
    if not _recog_lock.acquire(blocking=False):
        # Skip if previous frame is still being processed
        return
    
    try:
        b64_str = data.get('image', '')
        if not b64_str:
            emit('recognition_result', {'success': False, 'error': 'No image'})
            return
        
        # Decode image
        if ',' in b64_str:
            b64_str = b64_str.split(',')[1]
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            emit('recognition_result', {'success': False, 'error': 'Invalid image'})
            return
        
        svc = get_service()
        result = svc.recognize_realtime(img)
        emit('recognition_result', result)
        
    except Exception as e:
        logger.error(f"SocketIO recognize error: {e}")
        emit('recognition_result', {'success': False, 'error': str(e)})
    finally:
        _recog_lock.release()


# Lock for enrollment face check
_enroll_lock = threading.Lock()


@socketio.on('enroll_check_face')
def handle_enroll_check_face(data):
    """Check if face is properly positioned in oval for enrollment."""
    if not _enroll_lock.acquire(blocking=False):
        return  # Skip if still processing

    try:
        b64_str = data.get('image', '')
        if not b64_str:
            emit('enroll_face_status', {'face_ok': False, 'reason': 'no_image'})
            return

        if ',' in b64_str:
            b64_str = b64_str.split(',')[1]
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            emit('enroll_face_status', {'face_ok': False, 'reason': 'invalid_image'})
            return

        svc = get_service()
        
        # Use service's public API
        faces = svc.detect_faces(img)

        h, w = img.shape[:2]
        
        if not faces:
            logger.info(f"Enroll check: no face detected in {w}x{h} image")
            emit('enroll_face_status', {'face_ok': False, 'in_oval': False, 'reason': 'no_face'})
            return

        face = svc.get_center_face(faces, w, h)
        if not face:
            emit('enroll_face_status', {'face_ok': False, 'in_oval': False, 'reason': 'no_face'})
            return

        # Check if face is in oval
        in_oval = face.in_oval(w, h)

        # Check quality
        score, quality_ok, reason = face.quality_check(img)

        # Check distance
        dist_ok, dist_msg = face.distance_check(w)

        face_ok = in_oval and quality_ok and dist_ok
        
        logger.info(f"Enroll check: in_oval={in_oval}, quality={quality_ok}({reason}), dist={dist_ok}, score={score:.2f}, face_ok={face_ok}")

        emit('enroll_face_status', {
            'face_ok': face_ok,
            'in_oval': in_oval,
            'quality_ok': quality_ok,
            'quality_reason': reason,
            'dist_ok': dist_ok,
            'dist_msg': dist_msg if not dist_ok else '',
            'score': round(score, 2)
        })

    except Exception as e:
        logger.error(f"SocketIO enroll check error: {e}")
        emit('enroll_face_status', {'face_ok': False, 'reason': str(e)})
    finally:
        _enroll_lock.release()


# ==================== MAIN ====================

if __name__ == "__main__":
    logger.info("=" * 56)
    logger.info("FACE RECOGNITION SYSTEM v5.3 - SocketIO Realtime")
    logger.info("Flask + SocketIO + Supabase + MediaPipe + ArcFace")
    logger.info(f"Backend: {DB_BACKEND.upper()}")
    logger.info("=" * 56)
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase credentials not configured!")
        logger.warning("Set environment variables: SUPABASE_URL, SUPABASE_KEY")

    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=True, allow_unsafe_werkzeug=True)
