"""
FaceService v5.2 - Service Layer Facade Pattern (BUG-01 fix).
Decouples app.py (Web layer) from core/ modules.

app.py should ONLY call FaceService methods, never import detector/recognizer/db directly.
This protects the Web layer from interface changes in core/.
"""

import time
import threading
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
        # Locks cho thread-safe parallel processing
        self._detect_lock = threading.Lock()
        self._recog_lock = threading.Lock()
        self._match_lock = threading.Lock()
        
        # ByteTrack Tracker (thay Centroid Tracker)
        self._tracker = None
        try:
            from config import (TRACKER_ENABLED, TRACKER_MAX_LOST, 
                              TRACKER_IOU_THRESHOLD, TRACKER_HIGH_THRESH, TRACKER_MIN_HITS)
            if TRACKER_ENABLED:
                from models.tracker import ByteTracker
                self._tracker = ByteTracker(
                    max_lost=TRACKER_MAX_LOST,
                    iou_threshold=TRACKER_IOU_THRESHOLD,
                    high_thresh=TRACKER_HIGH_THRESH,
                    min_hits=TRACKER_MIN_HITS
                )
                logger.info("ByteTracker initialized (replacing Centroid Tracker)")
        except Exception as e:
            logger.warning(f"ByteTracker init failed: {e}. Using Centroid Tracker.")
        
        # Face Quality Assessor (thay rule-based)
        self._quality_assessor = None
        try:
            from models.quality import FaceQualityAssessor
            self._quality_assessor = FaceQualityAssessor()
            logger.info("FaceQualityAssessor initialized (Multi-Signal Scorer)")
        except Exception as e:
            logger.warning(f"FaceQualityAssessor init failed: {e}. Using rule-based quality.")
        
        logger.info("FaceService initialized")

    # ==================== QUALITY ASSESSMENT ====================

    def assess_quality(self, face, img_bgr):
        """Assess face quality using Multi-Signal Scorer (với 6DoF Pose).
        
        Returns (score, q_ok, q_reason) tương thích ngược với API cũ.
        """
        if self._quality_assessor is not None:
            # 1. Trích xuất ROI an toàn
            x, y, w, h = face.bbox
            fh, fw = img_bgr.shape[:2]
            y_start, y_end = max(0, y), min(fh, y + h)
            x_start, x_end = max(0, x), min(fw, x + w)
            face_roi_bgr = img_bgr[y_start:y_end, x_start:x_end]

            # 2. Gọi Multi-Signal Scorer (FaceQualityAssessor)
            score, details = self._quality_assessor.assess(
                face_roi_bgr=face_roi_bgr,
                landmarks_5pt=face.lm5,
                full_frame=img_bgr,
                bbox=face.bbox,
                lm2d=face.lm2d,  # Truyền lm2d để kích hoạt 6DoF Pose (solvePnP)
                img_w=fw,
                img_h=fh
            )

            # 3. Phân loại theo ngưỡng mới
            q_ok = self._quality_assessor.is_recognition_quality(score)
            q_reason = "OK" if q_ok else details['feedback']
            return score, q_ok, q_reason
        
        # Fallback: simple heuristic quality check
        return face.quality_check(img_bgr)
    # ==================== DETECTION ====================

    def detect_faces(self, img_bgr, tracking_state=None):
        """Detect faces in a BGR image. Thread-safe via lock.
        Returns list of FaceData objects.
        """
        import cv2
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with self._detect_lock:
            return self._detector.detect(rgb, tracking_state=tracking_state)

    def get_center_face(self, faces, frame_w, frame_h):
        """Get the face closest to center."""
        from models.detector import get_center_face
        return get_center_face(faces, frame_w, frame_h)

    def track_faces(self, faces):
        """Update ByteTracker with detected faces.
        
        Args:
            faces: list of FaceData objects from detect_faces()
            
        Returns:
            list of Track objects with .track_id, .bbox, .recognized_name
            Returns None if tracker not enabled (use centroid tracking)
        """
        if self._tracker is None:
            return None
        
        # Convert FaceData to detections: (x, y, w, h, confidence)
        detections = []
        for face in faces:
            x, y, w, h = face.bbox
            # Use det_score if available (SCRFD), else quality proxy
            conf = getattr(face, 'det_score', 0.9)
            detections.append((x, y, w, h, conf))
        
        return self._tracker.update(detections)

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
        
        q_score, q_ok, q_reason = self.assess_quality(face, img_bgr)
        
        # Reject khi quality quá tệ — bao gồm mờ, tối, nghiêng, rung
        # Không chỉ "Nho"/"Rong", mà bất kỳ failure nào có score < 0.3
        # đều cho embedding noise cao → match nhầm
        if not q_ok and (q_reason in {"Nho", "Rong"} or q_score < 0.3):
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
        
        # Bắt buộc rejected nếu MatchingEngine trả về Unknown (vd: do margin limit)
        if name == "Unknown":
            accepted = False
        else:
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

        q_score, ok, reason = self.assess_quality(face, img_bgr)
        
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
                # Sửa lỗi: Nếu là Unknown thì không bao giờ được accepted
                if name == "Unknown":
                    accepted = False
                else:
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
        
        Enhanced: Parallel frame processing via ThreadPoolExecutor.
        
        Args:
            images_bgr: list of BGR images
            threshold: (Optional) Override high-quality threshold
            
        Returns:
            dict with voting results
        """
        import concurrent.futures
        from config import THRESHOLD_ACCEPT_HIGH_QUALITY, THRESHOLD_ACCEPT_LOW_QUALITY, BLINK_EAR_CLOSED, BLINK_EAR_OPEN
        if threshold is None:
            threshold = THRESHOLD_ACCEPT_HIGH_QUALITY
        
        t_start = time.time()

        def _process_frame(args):
            """Xử lý 1 frame độc lập — thread-safe."""
            i, img = args
            if img is None:
                return {"frame": i, "status": "decode_error"}, None, None

            faces = self.detect_faces(img)
            h, w = img.shape[:2]
            face = self.get_center_face(faces, w, h)

            if face is None:
                return {"frame": i, "status": "no_face", "faces": len(faces)}, None, None

            q_score, q_ok, q_reason = self.assess_quality(face, img)
            left_ear, right_ear, avg_ear = face.eye_openness()

            hard_reject_reasons = {"Nho", "Rong"}
            if not q_ok and (q_reason in hard_reject_reasons or q_score < 0.3):
                return {"frame": i, "status": "quality_failed", "reason": q_reason}, None, None

            if self._spoofer:
                is_real, spoof_score = self._spoofer.is_real(img, face.bbox)
                if not is_real:
                    return {"frame": i, "status": "spoof_failed", "reason": "Spoof detected"}, None, None

            with self._recog_lock:
                emb = self._recognizer.get_embedding(img, face.lm5)
            if emb is None:
                return {"frame": i, "status": "embedding_failed"}, None, None

            emb = emb / np.linalg.norm(emb)
            with self._match_lock:
                name, score = self._db.match(emb)
            score = float(score)

            result = {
                "frame": i, "status": "ok",
                "name": name, "score": round(score, 4),
                "quality": round(float(q_score), 4) if q_ok else q_reason,
                "avg_ear": avg_ear
            }
            return result, emb, score

        # === XỬ LÝ SONG SONG TẤT CẢ FRAME ===
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(images_bgr), 4)) as pool:
            futures = list(pool.map(_process_frame, enumerate(images_bgr)))

        # Tách kết quả
        frame_results = []
        valid_embeddings = []
        valid_scores = []
        for fr, emb, score in futures:
            frame_results.append(fr)
            if fr["status"] == "ok" and emb is not None:
                valid_embeddings.append(emb)
                valid_scores.append(score)
                logger.info(f"FRAME {fr['frame']}: OK - match={fr['name']}, raw_score={score:.4f}")
            else:
                logger.info(f"FRAME {fr['frame']}: SKIP - {fr['status']}")

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

        # Majority name voting — loại "Unknown" ra khỏi voting
        # Vì Unknown không phải tên thật, nếu đếm sẽ làm lệch kết quả
        name_votes = {}
        for fr in frame_results:
            if fr.get("status") == "ok" and fr.get("name") and fr["name"] != "Unknown":
                n = fr["name"]
                name_votes[n] = name_votes.get(n, 0) + 1
        best_name = max(name_votes, key=name_votes.get) if name_votes else "Unknown"

        # Average embedding match
        avg_emb = np.mean(valid_embeddings, axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        avg_name, avg_emb_score = self._db.match(avg_emb)
        avg_emb_score = float(avg_emb_score)

        # Weighted average thay vì max() — tránh inflate score
        # Voting score (per-frame) đáng tin cậy hơn → 70%
        # Average embedding score bị triệt nhiễu nên cao hơn thực tế → 30%
        final_score = avg_score * 0.7 + avg_emb_score * 0.3
        
        # avg_name chỉ được override khi CÙNG TÊN với majority voting
        # Tránh trường hợp avg embedding match sang tên khác → nhận sai
        if avg_emb_score > avg_score and avg_name == best_name:
            final_name = avg_name
        else:
            final_name = best_name
        
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
                dynamic_reason = f"Low Quality + Blink OK (Threshold={THRESHOLD_ACCEPT_LOW_QUALITY})"
            else:
                applied_threshold = THRESHOLD_ACCEPT_HIGH_QUALITY
                dynamic_reason = f"Low Quality + No Blink (Threshold={THRESHOLD_ACCEPT_HIGH_QUALITY}) -> Needs Blink"
        else:
            dynamic_reason = f"High Quality (Threshold={THRESHOLD_ACCEPT_HIGH_QUALITY})"

        # Bắt buộc rớt nếu kết quả là Unknown
        final_accepted = (final_score >= applied_threshold) and (final_name != "Unknown")

        # HARD GATE: Liveness / Spoof
        spoof_count = sum(1 for fr in frame_results if fr.get("status") == "spoof_failed")
        total_processed = len([fr for fr in frame_results if fr.get("status") != "no_face"])
        if total_processed > 0 and (spoof_count / total_processed) >= 0.5:
            logger.warning(f"MULTI-FRAME LIVENESS BLOCKED: {spoof_count}/{total_processed} frames fake. Rejecting {best_name}.")
            final_accepted = False
            dynamic_reason = "Spoof detected (Hard Gate)"

        # BLINK CHECK: Chỉ cảnh báo, KHÔNG chặn cứng
        # Lý do: Multi-frame snapshot chụp 5 frame trong ~120ms → blink detection
        # cần tối thiểu ~300ms (open→close→open) → không thể phát hiện được.
        # Hard Gate blink chỉ hợp lý cho video stream liên tục (SocketIO realtime).
        if avg_q_score < 0.65 and not has_blink:
            logger.info(f"MULTI-FRAME BLINK NOTE: Quality low and no blink for {best_name}. (Warning only, not blocking)")
            if dynamic_reason not in ("Spoof detected (Hard Gate)",):
                dynamic_reason = f"Low Quality (Threshold={THRESHOLD_ACCEPT_HIGH_QUALITY})"

        if final_accepted:
            self._db.log_attendance(final_name, final_score)

        elapsed = time.time() - t_start
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
            "time_seconds": round(elapsed, 3)
        }

    # ==================== ENROLLMENT ====================

    def enroll_user(self, name, images_bgr):
        """Enroll a user from a list of BGR images.
        
        Returns:
            dict with enrollment results
        """
        from config import ENROLL_KEEP_TOP

        import concurrent.futures

        embeddings = []
        scores = []
        processed = 0
        skipped = 0

        def _process_enroll_frame(img):
            if img is None:
                return None

            # Mỗi frame được treat độc lập, tracking state riêng biệt
            faces = self.detect_faces(img, tracking_state=None)
            if len(faces) != 1:
                return None

            face = faces[0]
            q_score, ok, reason = self.assess_quality(face, img)
            if not ok:
                return None
                
            if self._spoofer:
                is_real, spoof_score = self._spoofer.is_real(img, face.bbox)
                if not is_real:
                    logger.warning(f"ENROLL REJECT - SPOOF DETECTED (score={spoof_score:.3f})")
                    return None

            emb = self._recognizer.get_embedding(img, face.lm5)
            if emb is not None:
                return (emb, q_score)
            return None

        # Xử lý đa luồng toàn bộ 30+ frames tải lên để tăng tốc enroll
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(_process_enroll_frame, images_bgr))

        for res in results:
            if res is not None:
                embeddings.append(res[0])
                scores.append(res[1])
                processed += 1
            else:
                skipped += 1

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
            "version": "5.6",
            "device": self._recognizer.device,
            "tta_enabled": TTA_ENABLED,
            "threshold": THRESHOLD,
            "total_users": len(users),
            "total_vectors": self._db.total,
            "database": "Supabase"
        }
