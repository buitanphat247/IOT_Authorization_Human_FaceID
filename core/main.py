"""
FACE RECOGNITION SYSTEM v5.1
OOP Architecture | MediaPipe + ArcFace + FAISS + SQLite
Enhanced: 2-Stage Threshold, Quality-Weighted Voting, 3-State Output,
          Prototype Matching, Enhanced Gating, A/B Test Framework,
          ONNX Optimized Inference, Multi-Threaded Pipeline, Score Stability.
"""

# ==================== DEPENDENCY CHECK (BUG-08 fix) ====================
# Không tự động pip install trong production.
# Chạy: pip install -r requirements.txt trước khi khởi động.
def _check_deps():
    """Kiểm tra thư viện cần thiết, CHỈ CẢNH BÁO nếu thiếu."""
    from logger import get_logger
    _logger = get_logger("deps")
    DEPS = [
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("mediapipe", "mediapipe"),
        ("onnxruntime", "onnxruntime"),
        ("faiss", "faiss-cpu"),
    ]
    missing = []
    for mod, pip_name in DEPS:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip_name)

    if missing:
        _logger.error(f"Thu vien thieu: {', '.join(missing)}")
        _logger.error("Chay: pip install -r requirements.txt")
        raise ImportError(f"Missing dependencies: {', '.join(missing)}")

_check_deps()
# ======================================================

import cv2
import numpy as np
import os, time, datetime
import threading
import concurrent.futures
from collections import defaultdict, deque

from config import *
from detector import FaceDetector
from recognizer import FaceRecognizer
from database import FaceDatabase
from anti_spoof import AntiSpoofer
from logger import get_logger

logger = get_logger("main")


# ==================== CAMERA THREAD (Tách luồng đọc frame) ====================

class CameraThread:
    """Luồng đọc camera riêng biệt, không block xử lý chính.
    
    Tại sao cần:
      - cv2.VideoCapture.read() bị chặn bởi driver camera (~30-50ms)
      - Trong lúc chờ camera, CPU nằm không → lãng phí
      - Tách luồng giúp luồng xử lý luôn có frame mới sẵn sàng
    
    BUG-07 fix: Dùng Lock + latest_frame thay vì Queue.
    Queue có race condition giữa full() và get_nowait() (TOCTOU).
    Lock + biến đơn = atomic, không bao giờ bị Empty exception.
    """
    
    def __init__(self, cap, queue_size=PIPELINE_QUEUE_SIZE):
        self.cap = cap
        self._lock = threading.Lock()
        self._latest_frame = None
        self._frame_ready = threading.Event()
        self._running = False
        self._thread = None
    
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self
    
    def _reader(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            # Ghi đè frame mới nhất (atomic với Lock)
            with self._lock:
                self._latest_frame = frame
            self._frame_ready.set()
    
    def read(self):
        """Lấy frame mới nhất (non-blocking, timeout=0.1s)."""
        if self._frame_ready.wait(timeout=0.1):
            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None
            self._frame_ready.clear()
            if frame is not None:
                return True, frame
        return False, None
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)


# ==================== SCORE STABILITY STATE MACHINE ====================

class ScoreStabilizer:
    """Bộ ổn định điểm số dùng EMA + Stability Check.
    
    Chống hiện tượng nhấp nháy kết quả bằng:
      1. EMA (Exponential Moving Average): làm mượt score liên frame
      2. Stability Check: chỉ chốt ACCEPT khi score ổn định N frame liên tiếp
    
    Tại sao cần:
      - Raw score từ FAISS dao động ±0.05 mỗi frame do noise
      - Người dùng thấy tên nhấp nháy ACCEPT/REJECT/ACCEPT → mất niềm tin
      - EMA + stability = score mượt + quyết định chắc chắn
    """
    
    def __init__(self):
        self._ema_scores = {}       # user_name -> EMA score hiện tại
        self._stable_count = {}     # user_name -> số frame liên tiếp ổn định
        self._last_status = {}      # user_name -> status cuối cùng
    
    def update(self, name, raw_score):
        """Cập nhật score qua bộ lọc EMA.
        
        Returns:
            (smoothed_score, is_stable)
        """
        if not SCORE_SMOOTHING_ENABLED:
            return raw_score, True
        
        alpha = SCORE_SMOOTHING_ALPHA
        
        # Lấy score EMA kỳ trước (nếu có)
        prev = self._ema_scores.get(name, None)
        
        if prev is not None:
            # Tính Exponential Moving Average
            smoothed = alpha * raw_score + (1 - alpha) * prev
            delta = abs(smoothed - prev)
        else:
            smoothed = raw_score
            delta = 0.0
        
        # LƯU LẠI state mới
        self._ema_scores[name] = smoothed
        
        # Stability check: dao động phải nhỏ hơn tolerance trong N frames liên tiếp
        if delta < SCORE_STABLE_TOLERANCE:
            self._stable_count[name] = self._stable_count.get(name, 0) + 1
        else:
            self._stable_count[name] = 0
        
        is_stable = self._stable_count.get(name, 0) >= SCORE_STABLE_FRAMES
        return smoothed, is_stable
    
    def reset(self, name=None):
        """Reset trạng thái (khi user biến mất khỏi camera)."""
        if name:
            self._ema_scores.pop(name, None)
            self._stable_count.pop(name, None)
            self._last_status.pop(name, None)
        else:
            self._ema_scores.clear()
            self._stable_count.clear()
            self._last_status.clear()


# ==================== MAIN APPLICATION ====================

class FaceApp:
    """Main application orchestrating detection, recognition, and UI."""

    def __init__(self):
        print("\n" + "=" * 56)
        print("   FACE RECOGNITION SYSTEM v5.2")
        print("   MediaPipe + ArcFace + FAISS + SQLite")
        print("   Optimized ONNX | Multi-Threaded Pipeline")
        print("   2-Stage Threshold | Score Stability EMA")
        print("=" * 56)

        if not os.path.exists(ARCFACE_PATH):
            logger.error(f"ArcFace not found: {ARCFACE_PATH}")
            raise FileNotFoundError(ARCFACE_PATH)

        t0 = time.time()
        logger.info("Loading models (parallel)...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            f_det = ex.submit(lambda: FaceDetector(mode="video"))
            f_rec = ex.submit(FaceRecognizer)
            f_cam = ex.submit(self._init_camera)
            f_spf = ex.submit(lambda: AntiSpoofer(model_path=ANTI_SPOOF_PATH, threshold=ANTI_SPOOF_THRESHOLD)) if os.path.exists(ANTI_SPOOF_PATH) else None

        self.detector = f_det.result()
        self.recognizer = f_rec.result()
        self.cap = f_cam.result()
        self.spoofer = f_spf.result() if f_spf else None
        self.db = FaceDatabase()
        self._fl_image = None
        self.stabilizer = ScoreStabilizer()

        if not self.cap.isOpened():
            raise RuntimeError("Camera not available!")

        logger.info(f"ArcFace: {self.recognizer.device}")
        logger.info(f"TTA: {'ON' if TTA_ENABLED else 'OFF'}")
        logger.info(f"Prototype: {'ON' if PROTOTYPE_ENABLED else 'OFF'}")
        logger.info(f"Pipeline Threaded: {'ON' if PIPELINE_THREADED else 'OFF'}")
        logger.info(f"Score EMA: {'ON' if SCORE_SMOOTHING_ENABLED else 'OFF'} (α={SCORE_SMOOTHING_ALPHA})")
        logger.info(f"2-Stage: REJECT<{THRESHOLD_REJECT} | ACCEPT>{THRESHOLD_ACCEPT}")
        logger.info(f"Ready in {time.time()-t0:.1f}s!")

    def _init_camera(self):
        cap = cv2.VideoCapture(CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
        return cap

    @property
    def fl_image(self):
        if self._fl_image is None:
            self._fl_image = FaceDetector(mode="image", num_faces=1)
        return self._fl_image

    # ==================== DRAWING UTILS ====================

    @staticmethod
    def draw_oval(frame, color=(200, 200, 200)):
        h, w = frame.shape[:2]
        cv2.ellipse(frame, (w//2, h//2), (int(w*0.18), int(h*0.30)), 0, 0, 360, color, 2)

    @staticmethod
    def draw_label(frame, text, bbox, color):
        bx, by, bw, bh = bbox
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), color, 2)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        cv2.rectangle(frame, (bx, by-25), (bx+tw+5, by), color, cv2.FILLED)
        cv2.putText(frame, text, (bx+2, by-5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)

    @staticmethod
    def draw_progress(frame, progress, y_offset=45, color=(0, 255, 0)):
        h, w = frame.shape[:2]
        bx, bw = int(w*0.1), int(w*0.8)
        cv2.rectangle(frame, (bx, h-y_offset), (bx+bw, h-y_offset+12), (50,50,50), -1)
        cv2.rectangle(frame, (bx, h-y_offset), (bx+int(bw*progress), h-y_offset+12), color, -1)

    # ==================== 2-STAGE DECISION + STABILITY ====================

    def decide_3state(self, score, q_ok, name="Unknown", is_stable=True):
        """Quyết định 3 trạng thái với stability check.
        
        Returns:
            (status, color)
        """
        if not q_ok:
            return "LOW_QUALITY", (128, 128, 128)
        
        if THREE_STATE_ENABLED:
            if score >= THRESHOLD_ACCEPT:
                # Chỉ chốt ACCEPT khi score ổn định
                if SCORE_SMOOTHING_ENABLED and not is_stable:
                    return "VERIFYING", (0, 200, 255)
                return "ACCEPT", (0, 255, 0)
            elif score <= THRESHOLD_REJECT:
                return "REJECT", (0, 0, 255)
            else:
                return "UNCERTAIN", (0, 200, 255)
        else:
            if score >= THRESHOLD:
                return "ACCEPT", (0, 255, 0)
            else:
                return "REJECT", (0, 0, 255)

    # ==================== QUALITY-WEIGHTED VOTING ====================

    @staticmethod
    def quality_weighted_score(score_history, quality_history):
        """Tính điểm trung bình có trọng số theo chất lượng frame."""
        if not score_history:
            return 0.0
        
        pairs = list(zip(score_history, quality_history))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = pairs[:MULTI_FRAME_TOP_N]
        
        if MULTI_FRAME_WEIGHTED:
            total_weight = sum(q for _, q in top_pairs)
            if total_weight == 0:
                return np.mean([s for s, _ in top_pairs])
            weighted = sum(s * q for s, q in top_pairs) / total_weight
            return float(weighted)
        else:
            return float(np.mean([s for s, _ in top_pairs]))

    # ==================== ENROLLMENT ====================

    def enroll_camera(self):
        print("\n  === DANG KY KHUON MAT ===")
        name = input("  Nhap ten: ").strip()
        if not name:
            return
        if name in self.db.get_users():
            if input(f"  '{name}' da co. Ghi de? (y/n): ").strip().lower() != 'y':
                return

        total_needed = sum(s[1] for s in ENROLL_STEPS)
        print(f"\n  Dang ky: {name} | {len(ENROLL_STEPS)} buoc | {total_needed} anh")
        print("  'q' de huy.\n")

        all_embs = []
        all_scores = []

        for step_i, (instruction, count, pose_dir) in enumerate(ENROLL_STEPS):
            step_embs = []
            step_scores = []
            last_t = 0

            while len(step_embs) < count:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                done = len(all_embs) + len(step_embs)
                self.draw_oval(frame, (0, 255, 0) if done > 0 else (200, 200, 200))

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.detector.detect(rgb)

                status, s_color = "", (255, 255, 255)

                if not faces:
                    status, s_color = "Khong thay khuon mat", (0, 0, 255)
                elif len(faces) > 1:
                    status, s_color = "Chi 1 nguoi truoc camera!", (0, 0, 255)
                else:
                    face = faces[0]
                    face.draw_mesh(frame)
                    bx, by, bw, bh = face.bbox

                    dist_ok, dist_msg = face.distance_check(w)
                    if not dist_ok:
                        status, s_color = dist_msg, (0, 165, 255)
                        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 165, 255), 2)
                    elif not face.in_oval(w, h):
                        status, s_color = "Di chuyen mat vao GIUA khung", (0, 165, 255)
                        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 165, 255), 2)
                    else:
                        pose_ok, pose_msg = face.check_pose(pose_dir)
                        if not pose_ok:
                            status, s_color = pose_msg, (255, 0, 255)
                            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (255, 0, 255), 2)
                        else:
                            q_score, q_ok, q_reason = face.quality_check(frame)
                            if not q_ok:
                                status, s_color = f"Chat luong: {q_reason}", (0, 0, 255)
                                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 0, 255), 2)
                            else:
                                # Liveness Check
                                spf_ok = True
                                if self.spoofer:
                                    is_real, spf_score = self.spoofer.is_real(frame, face.bbox)
                                    if not is_real:
                                        spf_ok = False
                                        status = f"Giả mạo! ({spf_score:.2f})"
                                        s_color = (0, 0, 255)
                                        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 0, 255), 2)
                                        
                                if spf_ok:
                                    if time.time() - last_t >= ENROLL_INTERVAL:
                                        emb = self.recognizer.get_embedding(frame, face.lm5)
                                        if emb is not None:
                                            step_embs.append(emb)
                                            step_scores.append(q_score)
                                            last_t = time.time()
                                            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0,255,0), 4)
                                            status = f"DA CHUP! ({len(step_embs)}/{count})"
                                            s_color = (0, 255, 0)
                                    else:
                                        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0,255,0), 2)
                                        status, s_color = "Tot! Giu nguyen...", (0, 255, 0)

                # HUD
                done = len(all_embs) + len(step_embs)
                cv2.putText(frame, f"Buoc {step_i+1}/{len(ENROLL_STEPS)}: {instruction}",
                            (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255,255,255), 2)
                cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, s_color, 1)
                cv2.putText(frame, f"User: {name}", (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
                self.draw_progress(frame, done / total_needed, 45, (0, 255, 0))
                self.draw_progress(frame, len(step_embs) / count, 25, (0, 255, 255))
                cv2.putText(frame, f"{done}/{total_needed}", (int(w*0.9)+5, h-35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

                cv2.imshow("Face Recognition System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Enrollment cancelled by user")
                    cv2.destroyAllWindows()
                    return

            all_embs.extend(step_embs)
            all_scores.extend(step_scores)
            logger.info(f"Step {step_i+1} done: {instruction} ({len(step_embs)} images)")

        cv2.destroyAllWindows()

        if len(all_embs) >= total_needed:
            final_embs, final_scores, prototype = self.recognizer.select_best_embeddings(
                all_embs, all_scores, keep_top=ENROLL_KEEP_TOP
            )
            self.db.add_user(name, final_embs, final_scores, prototype)
            logger.info(f"ENROLLED: '{name}' | {len(all_embs)} captured -> {len(final_embs)} saved (top-{ENROLL_KEEP_TOP}) | FAISS: {self.db.total} vec")
            if prototype is not None:
                logger.info("Prototype vector: GENERATED")

    # ==================== FOLDER ENROLLMENT ====================

    def enroll_folder(self):
        print(f"\n  === DANG KY TU: {DATA_DIR} ===")
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR, exist_ok=True)
            print(f"  Them anh vao {DATA_DIR}/<ten>/ roi chay lai.")
            return

        dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
        if not dirs:
            print("  Khong co thu muc user.")
            return

        det = self.fl_image
        for user in dirs:
            imgs = [f for f in os.listdir(os.path.join(DATA_DIR, user))
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not imgs:
                continue
            print(f"  {user}: {len(imgs)} anh...")
            embs = []
            for fn in imgs:
                img = cv2.imread(os.path.join(DATA_DIR, user, fn))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = det.detect(rgb)
                if len(faces) != 1:
                    continue
                face = faces[0]
                _, ok, _ = face.quality_check(img)
                if not ok:
                    continue
                emb = self.recognizer.get_embedding(img, face.lm5)
                if emb is not None:
                    embs.append(emb)
            if embs:
                cleaned = self.recognizer.clean_embeddings(embs)
                prototype = self.recognizer.compute_prototype(cleaned)
                self.db.add_user(user, cleaned, prototype=prototype)
                logger.info(f"{user}: {len(cleaned)} emb (filtered from {len(embs)}) + Prototype")

        logger.info(f"DONE: FAISS: {self.db.total} vec | {len(self.db.get_users())} users")

    # ==================== VERIFY MODE (Multi-Threaded Pipeline) ====================

    def verify_mode(self):
        users = self.db.get_users()
        if not users:
            print("\n  DB trong! Dang ky truoc.")
            return

        print(f"\n  === NHAN DIEN 24/7 | {len(users)} users | {self.db.total} vec ===")
        print(f"  ArcFace: {self.recognizer.device} | TTA: {'ON' if TTA_ENABLED else 'OFF'}")
        print(f"  Pipeline: {'THREADED' if PIPELINE_THREADED else 'SINGLE'} | EMA: α={SCORE_SMOOTHING_ALPHA}")
        print(f"  Threshold: REJECT<{THRESHOLD_REJECT} | ACCEPT>{THRESHOLD_ACCEPT}")
        print(f"  'r' ghi hinh | 'q' thoat")

        prev_t = 0
        recording, writer = False, None
        os.makedirs(RECORDS_DIR, exist_ok=True)

        # Quality-Weighted Multi-Frame Voting State
        score_history = defaultdict(lambda: deque(maxlen=MULTI_FRAME_BUFFER))
        quality_history = defaultdict(lambda: deque(maxlen=MULTI_FRAME_BUFFER))
        
        # Face-gone Tracking (chặn contaminated voting)
        last_seen = {}
        UNSEEN_TIMEOUT = 3.0  # seconds
        
        # Score Stability State Machine
        self.stabilizer.reset()

        # Camera Thread (tách luồng đọc frame)
        cam_thread = None
        if PIPELINE_THREADED:
            cam_thread = CameraThread(self.cap, PIPELINE_QUEUE_SIZE).start()
            logger.info("Camera thread started")

        # Frame skip counter (Ưu Tiên 5: Tăng tốc)
        frame_counter = 0
        cached_results = {}  # face_id -> (label, status, color, raw_score, q_score)

        try:
            while True:
                # Đọc frame từ camera thread hoặc trực tiếp
                if PIPELINE_THREADED and cam_thread:
                    ret, frame = cam_thread.read()
                else:
                    ret, frame = self.cap.read()

                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                now = time.time()
                fps = 1 / (now - prev_t) if prev_t > 0 else 0
                prev_t = now

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.detector.detect(rgb)

                # Frame skip: chỉ chạy recognize mỗi (FRAME_SKIP+1) frame
                do_recognize = (frame_counter % (FRAME_SKIP + 1) == 0) if FRAME_SKIP > 0 else True
                frame_counter += 1

                for face in faces:
                    face.draw_mesh(frame)
                    bx, by, bw, bh = face.bbox
                    q_score, q_ok, reason = face.quality_check(frame)

                    if not q_ok:
                        status, color = "LOW_QUALITY", (128, 128, 128)
                        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), color, 1)
                        cv2.putText(frame, f"{reason}", (bx, by-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    else:
                        # Liveness Check
                        spf_ok = True
                        if self.spoofer:
                            is_real, spf_score = self.spoofer.is_real(frame, face.bbox)
                            if not is_real:
                                spf_ok = False
                                status, color = f"SPOOF ({spf_score:.2f})", (0, 0, 255)
                                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), color, 2)
                                cv2.putText(frame, status, (bx, by-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)
                        
                        if spf_ok:
                            face_id = f"{bx//50}_{by//50}"
                            
                            if do_recognize:
                                emb = self.recognizer.get_embedding(frame, face.lm5)
                                if emb is not None:
                                    # 1. Database match (prototype + top-K)
                                    name, raw_score = self.db.match(emb)
                                    
                                    # === FIX: Raw score floor — REJECT ngay nếu quá thấp ===
                                    if raw_score < THRESHOLD_REJECT:
                                        name = "Unknown"
                                    
                                    # === FIX: Phát hiện đổi mặt — Reset khi tên thay đổi ===
                                    prev_name = getattr(self, '_prev_face_names', {}).get(face_id)
                                    if not hasattr(self, '_prev_face_names'):
                                        self._prev_face_names = {}
                                    
                                    if prev_name and prev_name != name and name != "Unknown":
                                        score_history.pop(prev_name, None)
                                        quality_history.pop(prev_name, None)
                                        self.stabilizer.reset(prev_name)
                                        last_seen.pop(prev_name, None)
                                        logger.info(f"Face change detected: {prev_name} -> {name}. History reset.")
                                    
                                    self._prev_face_names[face_id] = name
                                    
                                    # 2. Quality-Weighted Multi-frame Voting
                                    if MULTI_FRAME_ENABLED and name != "Unknown":
                                        last_seen[name] = now
                                        score_history[name].append(raw_score)
                                        quality_history[name].append(q_score)
                                        voted_score = self.quality_weighted_score(
                                            list(score_history[name]),
                                            list(quality_history[name])
                                        )
                                    else:
                                        voted_score = raw_score
                                    
                                    # 3. Score Stability EMA (Temporal Smoothing)
                                    smoothed_score, is_stable = self.stabilizer.update(name, voted_score)
                                    final_score = smoothed_score
                                        
                                    # 4. 2-Stage Decision with Stability
                                    status, color = self.decide_3state(final_score, q_ok, name, is_stable)
                                    
                                    if status == "ACCEPT":
                                        label = f"{name} ({final_score:.2f})"
                                    elif status in ("UNCERTAIN", "VERIFYING"):
                                        label = f"?{name}? ({final_score:.2f})"
                                    else:
                                        label = f"Unknown ({final_score:.2f})"

                                    # Cache kết quả cho frame skip
                                    cached_results[face_id] = (label, status, color, raw_score, q_score)
                                    
                                    self.draw_label(frame, label, face.bbox, color)
                                    cv2.putText(frame, status, (bx, by+bh+18),
                                                cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
                                    cv2.putText(frame, f"Q:{q_score:.0%} R:{raw_score:.2f}", (bx, by+bh+35),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)
                            else:
                                # Frame skip: dùng cached result nếu có
                                cached = cached_results.get(face_id)
                                if cached:
                                    label, status, color, raw_score, q_score = cached
                                    self.draw_label(frame, label, face.bbox, color)
                                    cv2.putText(frame, status, (bx, by+bh+18),
                                                cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
                                    cv2.putText(frame, f"Q:{q_score:.0%} R:{raw_score:.2f}", (bx, by+bh+35),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

                # Memory Cleanup: Dọn dẹp score buffer của các khuôn mặt đã rời camera
                for uname in list(last_seen.keys()):
                    if now - last_seen[uname] > UNSEEN_TIMEOUT:
                        score_history.pop(uname, None)
                        quality_history.pop(uname, None)
                        self.stabilizer.reset(uname)
                        del last_seen[uname]

                # HUD
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                pipeline_tag = "THREADED" if PIPELINE_THREADED else "SINGLE"
                cv2.putText(frame, f"{self.recognizer.device} | {pipeline_tag} | FAISS:{self.db.total}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200,200,200), 1)
                cv2.putText(frame, f"Accept>{THRESHOLD_ACCEPT} Reject<{THRESHOLD_REJECT} EMA:a={SCORE_SMOOTHING_ALPHA}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150,150,150), 1)

                if recording:
                    cv2.circle(frame, (w-30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(frame, "REC", (w-70, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    if writer:
                        writer.write(frame)

                cv2.imshow("Face Recognition System", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if not recording:
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        writer = cv2.VideoWriter(
                            os.path.join(RECORDS_DIR, f"rec_{ts}.avi"),
                            cv2.VideoWriter_fourcc(*'XVID'), 20.0, (w, h)
                        )
                        recording = True
                        logger.info("Recording started")
                    else:
                        recording = False
                        if writer:
                            writer.release()
                            writer = None
                        logger.info("Recording stopped")

        finally:
            if cam_thread:
                cam_thread.stop()
                logger.info("Camera thread stopped")
            if writer:
                writer.release()

    # ==================== MENU ====================

    def run(self):
        try:
            while True:
                u = self.db.get_users()
                print("\n" + "=" * 56)
                print(f"  MENU | FAISS: {self.db.total} vec | {len(u)} users")
                print(f"  Engine: {self.recognizer.device}")
                print("=" * 56)
                for n, c in u.items():
                    proto_tag = " [P]" if self.db.has_prototype(n) else ""
                    print(f"    - {n}: {c} emb{proto_tag}")
                print("\n  1. DANG KY (Camera)")
                print("  2. DANG KY (Thu muc)")
                print("  3. NHAN DIEN 24/7")
                print("  4. XOA user")
                print("  0. THOAT\n")

                ch = input("  Chon: ").strip()

                if ch == '1':
                    self.enroll_camera()
                elif ch == '2':
                    self.enroll_folder()
                elif ch == '3':
                    self.verify_mode()
                elif ch == '4':
                    self._delete_user()
                elif ch == '0':
                    break

        except KeyboardInterrupt:
            logger.info("Shutting down (Ctrl+C)")

        self.cleanup()

    def _delete_user(self):
        users_list = list(self.db.get_users().items())
        if not users_list:
            print("  DB trong.")
            return

        print("\n  --- DANH SACH USER ---")
        for i, (n, c) in enumerate(users_list):
            proto_tag = " [P]" if self.db.has_prototype(n) else ""
            print(f"    [{i+1}] {n}  ({c} emb{proto_tag})")
        print(f"    [0] Huy")

        try:
            sel = int(input(f"\n  Chon (0-{len(users_list)}): ").strip())
            if 1 <= sel <= len(users_list):
                name = users_list[sel - 1][0]
                if input(f"  Xoa '{name}'? (y/n): ").strip().lower() == 'y':
                    self.db.remove_user(name)
                    logger.info(f"DELETED: '{name}'")
                else:
                    print("  [HUY]")
            elif sel != 0:
                print("  So khong hop le.")
        except ValueError:
            print("  Nhap so.")

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.db.close()
        self.detector.close()
        if self._fl_image:
            self._fl_image.close()
        logger.info("System shutdown complete")


# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    try:
        app = FaceApp()
        app.run()
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Thoat.")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        cv2.destroyAllWindows()
