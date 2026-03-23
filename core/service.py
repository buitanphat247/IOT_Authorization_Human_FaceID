"""
FaceService v5.2 - Service Layer Facade Pattern (BUG-01 fix).
Decouples app.py (Web layer) from core/ modules.

app.py should ONLY call FaceService methods, never import detector/recognizer/db directly.
This protects the Web layer from interface changes in core/.
"""

import time
import numpy as np
from logger import get_logger

logger = get_logger("service")


class FaceService:
    """Facade for the face recognition pipeline.
    
    All business logic passes through here.
    Web layer (app.py) calls these methods only.
    """

    def __init__(self, detector, recognizer, db, spoofer=None):
        self._detector = detector
        self._recognizer = recognizer
        self._db = db
        self._spoofer = spoofer
        logger.info("FaceService initialized")

    # ==================== DETECTION ====================

    def detect_faces(self, img_bgr, tracking_state=None):
        """Detect faces in a BGR image.
        Returns list of FaceData objects.
        """
        import cv2
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self._detector.detect(rgb, tracking_state=tracking_state)

    def get_center_face(self, faces, frame_w, frame_h):
        """Get the face closest to center."""
        from detector import get_center_face
        return get_center_face(faces, frame_w, frame_h)

    # ==================== REALTIME RECOGNITION (SocketIO optimized) ====================

    def recognize_realtime(self, img_bgr):
        """Ultra-fast single-frame recognition for SocketIO realtime.
        
        Optimized for speed:
          - Single frame, no multi-frame voting
          - Minimal overhead, immediate response
          - Anti-spoof check included
          
        Returns:
            dict with: success, name, accepted, status
        """
        from config import THRESHOLD_ACCEPT_HIGH_QUALITY, THRESHOLD_REJECT
        
        faces = self.detect_faces(img_bgr)
        h, w = img_bgr.shape[:2]
        face = self.get_center_face(faces, w, h)
        
        if face is None:
            return {
                "success": True,
                "name": "",
                "accepted": False,
                "status": "no_face"
            }
        
        q_score, q_ok, q_reason = face.quality_check(img_bgr)
        
        # Skip extremely bad quality
        if not q_ok and q_reason in {"Nho", "Rong"}:
            return {
                "success": True,
                "name": "",
                "accepted": False,
                "status": "low_quality"
            }
        
        # Anti-Spoof check
        if self._spoofer:
            is_real, spf_score = self._spoofer.is_real(img_bgr, face.bbox)
            if not is_real:
                return {
                    "success": True,
                    "name": "",
                    "accepted": False,
                    "status": "spoof"
                }
        
        # Get embedding and match
        emb = self._recognizer.get_embedding(img_bgr, face.lm5)
        if emb is None:
            return {
                "success": True,
                "name": "",
                "accepted": False,
                "status": "no_embedding"
            }
        
        name, score = self._db.match(emb)
        score = float(score)
        accepted = score >= THRESHOLD_ACCEPT_HIGH_QUALITY
        
        if score < THRESHOLD_REJECT:
            name = "Unknown"
        
        display_name = name if accepted else "Unknown"
        
        if accepted:
            self._db.log_attendance(name, score)
            logger.info(f"REALTIME ACCEPT: {name} (score={score:.4f})")
        
        return {
            "success": True,
            "name": display_name,
            "accepted": accepted,
            "status": "accepted" if accepted else "rejected"
        }

    # ==================== RECOGNITION ====================

    def recognize_single(self, img_bgr, threshold=None):
        """Recognize a face from a single image.
        
        Returns:
            dict with keys: success, name, score, accepted, quality_score, bbox, etc.
        """
        from config import THRESHOLD_ACCEPT_HIGH_QUALITY
        if threshold is None:
            threshold = THRESHOLD_ACCEPT_HIGH_QUALITY
        faces = self.detect_faces(img_bgr)
        h, w = img_bgr.shape[:2]
        face = self.get_center_face(faces, w, h)

        if face is None:
            return {
                "success": True,
                "faces_detected": len(faces),
                "results": []
            }

        q_score, ok, reason = face.quality_check(img_bgr)
        
        # --- Liveness Detection (Anti-Spoofing) ---
        if ok and self._spoofer:
            is_real, spf_score = self._spoofer.is_real(img_bgr, face.bbox)
            if not is_real:
                ok = False
                reason = f"Spoof ({(1-spf_score)*100:.1f}%)"

        result = {
            "name": "Unknown",
            "score": 0,
            "accepted": False,
            "quality_score": round(float(q_score), 4),
            "bbox": [int(x) for x in face.bbox],
        }

        if not ok:
            result["quality"] = reason
        else:
            emb = self._recognizer.get_embedding(img_bgr, face.lm5)
            if emb is not None:
                name, score = self._db.match(emb)
                score = float(score)
                accepted = bool(score >= threshold)
                result["name"] = name if accepted else "Unknown"
                result["score"] = round(score, 4)
                result["accepted"] = accepted

                if accepted:
                    self._db.log_attendance(name, score)
                    logger.info(f"RECOGNIZED: {name} (score={score:.4f})")

        return {
            "success": True,
            "faces_detected": len(faces),
            "results": [result]
        }

    def recognize_multi(self, images_bgr, threshold=None):
        """Multi-frame voting recognition with Dynamic Threshold.
        
        Args:
            images_bgr: list of BGR images
            threshold: (Optional) Override high-quality threshold
            
        Returns:
            dict with voting results
        """
        from config import THRESHOLD_ACCEPT_HIGH_QUALITY, THRESHOLD_ACCEPT_LOW_QUALITY, BLINK_EAR_CLOSED, BLINK_EAR_OPEN
        if threshold is None:
            threshold = THRESHOLD_ACCEPT_HIGH_QUALITY
        valid_embeddings = []
        valid_scores = []
        frame_results = []
        tracking_state = {}

        for i, img in enumerate(images_bgr):
            if img is None:
                logger.info(f"FRAME {i}: SKIP - decode_error")
                frame_results.append({"frame": i, "status": "decode_error"})
                continue

            faces = self.detect_faces(img, tracking_state=tracking_state)
            h, w = img.shape[:2]
            face = self.get_center_face(faces, w, h)

            if face is None:
                logger.info(f"FRAME {i}: SKIP - no_face (detected={len(faces)})")
                frame_results.append({"frame": i, "status": "no_face", "faces": len(faces)})
                continue

            q_score, q_ok, q_reason = face.quality_check(img)
            left_ear, right_ear, avg_ear = face.eye_openness()
            logger.info(f"FRAME {i}: quality={q_score:.3f}, ok={q_ok}, reason={q_reason}, ear={avg_ear:.3f}")
            
            # Cho multi-frame API: chỉ HARD REJECT khi face quá nhỏ hoặc rỗng
            # Các lỗi quality khác (blur, brightness, tilt) → vẫn tiếp tục extract embedding
            # vì multi-frame voting sẽ giảm trọng số frame chất lượng thấp
            hard_reject_reasons = {"Nho", "Rong"}
            if not q_ok and q_reason in hard_reject_reasons:
                logger.info(f"FRAME {i}: SKIP - hard reject ({q_reason})")
                frame_results.append({"frame": i, "status": "quality_failed", "reason": q_reason})
                continue
            
            # Anti-Spoofing detection using the new v2 model
            if self._spoofer:
                is_real, spoof_score = self._spoofer.is_real(img, face.bbox)
                if not is_real:
                    logger.warning(f"FRAME {i}: REJECT - SPOOF DETECTED (score={spoof_score:.3f})")
                    frame_results.append({"frame": i, "status": "spoof_failed", "reason": "Spoof detected"})
                    continue
                
            emb = self._recognizer.get_embedding(img, face.lm5)

            if emb is None:
                logger.info(f"FRAME {i}: SKIP - embedding_failed")
                frame_results.append({"frame": i, "status": "embedding_failed"})
                continue

            emb = emb / np.linalg.norm(emb)
            name, score = self._db.match(emb)
            score = float(score)
            
            logger.info(f"FRAME {i}: OK - match={name}, raw_score={score:.4f}")

            valid_embeddings.append(emb)
            valid_scores.append(score)

            frame_results.append({
                "frame": i, "status": "ok",
                "name": name, "score": round(score, 4),
                "quality": round(float(q_score), 4) if q_ok else q_reason,
                "avg_ear": avg_ear
            })

        if not valid_scores:
            return {
                "success": True, "recognized": False, "name": "Unknown",
                "avg_score": 0, "reason": "No valid faces",
                "frames_total": len(images_bgr), "frames_valid": 0,
                "frame_details": frame_results
            }

        # Multi-frame voting: top 70% scores
        sorted_scores = sorted(valid_scores, reverse=True)
        keep_count = max(2, int(len(sorted_scores) * 0.7))
        avg_score = float(np.mean(sorted_scores[:keep_count]))

        # Majority name voting
        name_votes = {}
        for fr in frame_results:
            if fr.get("status") == "ok" and fr.get("name"):
                n = fr["name"]
                name_votes[n] = name_votes.get(n, 0) + 1
        best_name = max(name_votes, key=name_votes.get) if name_votes else "Unknown"

        # Average embedding match
        avg_emb = np.mean(valid_embeddings, axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        avg_name, avg_emb_score = self._db.match(avg_emb)
        avg_emb_score = float(avg_emb_score)

        final_score = max(avg_score, avg_emb_score)
        final_name = avg_name if avg_emb_score > avg_score else best_name
        
        # --- DYNAMIC THRESHOLD LOGIC ---
        # Tính quality trung bình của tất cả valid frames
        valid_q_scores = [fr["quality"] for fr in frame_results if fr.get("status") == "ok" and isinstance(fr.get("quality"), float)]
        avg_q_score = np.mean(valid_q_scores) if valid_q_scores else 0.0
        
        # Kiểm tra Blink trong chuỗi frame
        ears = [fr["avg_ear"] for fr in frame_results if fr.get("status") == "ok" and "avg_ear" in fr]
        min_ear = min(ears) if ears else 1.0
        max_ear = max(ears) if ears else 0.0
        has_blink = (min_ear <= BLINK_EAR_CLOSED) and (max_ear >= BLINK_EAR_OPEN)
        
        applied_threshold = threshold
        dynamic_reason = "High Quality (No Blink Req)"
        
        # Nếu chất lượng thấp (mờ, tối) < 0.65 -> Hạ ngưỡng nhưng yêu cầu chớp mắt
        if avg_q_score < 0.65:
            if has_blink:
                applied_threshold = THRESHOLD_ACCEPT_LOW_QUALITY
                dynamic_reason = "Low Quality + Blink OK (Threshold=0.58)"
            else:
                applied_threshold = THRESHOLD_ACCEPT_HIGH_QUALITY
                dynamic_reason = "Low Quality + No Blink (Threshold=0.62) -> Needs Blink"
        else:
            dynamic_reason = "High Quality (Threshold=0.62)"

        final_accepted = final_score >= applied_threshold

        # --- KIỂM DUYỆT LIVENESS TỔNG THỂ ---
        # Ngăn chặn trường hợp 5 khung hình đưa điện thoại vào, 
        # model lọt lưới 1 khung hình qua mặt được Anti-Spoof nhưng 4 khung hình còn lại bị bắt lỗi giả mạo.
        # Nếu chỉ dựa vào "1 khung lọt lưới" thì face_match vẫn đúng -> hacker qua cổng.
        # SỬA: Nếu tỷ lệ khung hình bị đánh dấu giả mạo >= 40% -> ĐÁNH TRƯỢT TOÀN BỘ PHIÊN!
        spoof_count = sum(1 for fr in frame_results if fr.get("status") == "spoof_failed")
        if len(images_bgr) > 0 and (spoof_count / len(images_bgr)) >= 0.4:
            logger.warning(f"MULTI-FRAME LIVENESS BLOCKED: {spoof_count}/{len(images_bgr)} frames fake. Rejecting {best_name}.")
            final_accepted = False

        if final_accepted:
            self._db.log_attendance(final_name, final_score)

        elapsed = time.time() - (time.time())  # placeholder
        logger.info(f"MULTI-FRAME: {len(valid_scores)}/{len(images_bgr)} valid | "
                     f"{final_name} score={final_score:.4f} throbj={applied_threshold:.2f} "
                     f"[{dynamic_reason}] {'ACCEPTED' if final_accepted else 'REJECTED'}")

        # Nếu REJECTED → trả "Unknown", không lộ tên người dùng trong DB
        display_name = final_name if final_accepted else "Unknown"

        return {
            "success": True, "recognized": final_accepted,
            "name": display_name,
            "score": round(final_score, 4),
            "avg_score_voting": round(avg_score, 4),
            "avg_score_embedding": round(avg_emb_score, 4),
            "accepted": final_accepted,
            "dynamic_threshold": applied_threshold,
            "dynamic_reason": dynamic_reason,
            "has_blink": has_blink,
            "frames_total": len(images_bgr), "frames_valid": len(valid_scores),
            "per_frame_scores": [round(s, 4) for s in valid_scores],
            "frame_details": frame_results,
            "method": "multi_frame_voting",
            "time_seconds": "N/A"
        }

    # ==================== ENROLLMENT ====================

    def enroll_user(self, name, images_bgr):
        """Enroll a user from a list of BGR images.
        
        Returns:
            dict with enrollment results
        """
        from config import ENROLL_KEEP_TOP

        embeddings = []
        scores = []
        processed = 0
        skipped = 0
        tracking_state = {}

        for img in images_bgr:
            if img is None:
                skipped += 1
                continue

            faces = self.detect_faces(img, tracking_state=tracking_state)
            if len(faces) != 1:
                skipped += 1
                continue

            face = faces[0]
            q_score, ok, reason = face.quality_check(img)
            if not ok:
                skipped += 1
                continue
                
            if self._spoofer:
                is_real, spoof_score = self._spoofer.is_real(img, face.bbox)
                if not is_real:
                    logger.warning(f"ENROLL REJECT - SPOOF DETECTED (score={spoof_score:.3f})")
                    skipped += 1
                    continue

            emb = self._recognizer.get_embedding(img, face.lm5)
            if emb is not None:
                embeddings.append(emb)
                scores.append(q_score)
                processed += 1

        if len(embeddings) < 3:
            return {
                "success": False,
                "error": f"Only {len(embeddings)} valid faces found. Need at least 3.",
                "processed": processed, "skipped": skipped
            }

        final_embs, final_scores, prototype = self._recognizer.select_best_embeddings(
            embeddings, scores, keep_top=ENROLL_KEEP_TOP
        )
        self._db.add_user(name, final_embs, scores=final_scores, prototype=prototype)

        logger.info(f"ENROLLED: {name} ({len(final_embs)} embeddings)")

        return {
            "success": True,
            "message": f"User '{name}' enrolled successfully",
            "total_embeddings": len(final_embs),
            "processed": processed, "skipped": skipped,
            "quality_range": f"{min(final_scores):.3f} ~ {max(final_scores):.3f}" if final_scores else "N/A"
        }

    # ==================== POSE CHECK ====================

    def check_pose(self, img_bgr, direction="straight"):
        """Check head pose direction."""
        faces = self.detect_faces(img_bgr)
        if len(faces) != 1:
            return {"valid": False, "reason": "no_face" if not faces else "multiple_faces"}

        face = faces[0]
        h_off, v_ratio = face.head_pose()
        pose_ok, pose_msg = face.check_pose(direction)

        return {
            "valid": bool(pose_ok),
            "h_offset": round(float(h_off), 3),
            "v_ratio": round(float(v_ratio), 3),
            "direction": direction,
            "message": pose_msg
        }

    # ==================== DATA ACCESS ====================

    def get_users(self):
        return self._db.get_users()

    def delete_user(self, name):
        self._db.remove_user(name)
        logger.info(f"DELETED user: {name}")

    def sync_db(self):
        if hasattr(self._db, 'sync_from_supabase'):
            self._db.sync_from_supabase()

    def get_attendance_logs(self, limit=50):
        return self._db.get_attendance_logs(limit=limit)

    def get_system_info(self):
        from config import TTA_ENABLED, THRESHOLD
        users = self._db.get_users()
        return {
            "version": "5.2",
            "device": self._recognizer.device,
            "tta_enabled": TTA_ENABLED,
            "threshold": THRESHOLD,
            "total_users": len(users),
            "total_vectors": self._db.total,
            "database": "Supabase"
        }
