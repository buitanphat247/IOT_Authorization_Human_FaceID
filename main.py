"""
FACE RECOGNITION SYSTEM v4.0
OOP Architecture | MediaPipe + ArcFace + FAISS + SQLite
Accuracy: TTA + Outlier Removal + Quality Scoring + Head Pose Validation
"""

# ==================== AUTO INSTALL ====================
def _ensure_deps():
    """Tu dong cai thu vien thieu khi chay lan dau."""
    import subprocess, sys
    # (import_name, pip_name)
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
        print(f"[AUTO] Dang cai: {', '.join(missing)}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )
        print("[AUTO] Xong!")

_ensure_deps()
# ======================================================

import cv2
import numpy as np
import os, time, datetime
import concurrent.futures

from config import *
from detector import FaceDetector
from recognizer import FaceRecognizer
from database import FaceDatabase


class FaceApp:
    """Main application orchestrating detection, recognition, and UI."""

    def __init__(self):
        print("\n" + "=" * 56)
        print("   FACE RECOGNITION SYSTEM v4.0")
        print("   MediaPipe + ArcFace + FAISS + SQLite")
        print("   TTA | Outlier Removal | Head Pose Validation")
        print("=" * 56)

        if not os.path.exists(ARCFACE_PATH):
            print(f"[LOI] ArcFace not found: {ARCFACE_PATH}")
            raise FileNotFoundError(ARCFACE_PATH)

        t0 = time.time()
        print("  Loading models (parallel)...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            f_det = ex.submit(lambda: FaceDetector(mode="video"))
            f_rec = ex.submit(FaceRecognizer)
            f_cam = ex.submit(self._init_camera)

        self.detector = f_det.result()
        self.recognizer = f_rec.result()
        self.cap = f_cam.result()
        self.db = FaceDatabase()
        self._fl_image = None  # lazy load

        if not self.cap.isOpened():
            raise RuntimeError("Camera not available!")

        print(f"    -> ArcFace: {self.recognizer.device}")
        print(f"    -> TTA: {'ON' if TTA_ENABLED else 'OFF'}")
        print(f"  [OK] Ready in {time.time()-t0:.1f}s!\n")

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

                    # 1. Distance check
                    dist_ok, dist_msg = face.distance_check(w)
                    if not dist_ok:
                        status, s_color = dist_msg, (0, 165, 255)
                        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 165, 255), 2)

                    # 2. Oval check
                    elif not face.in_oval(w, h):
                        status, s_color = "Di chuyen mat vao GIUA khung", (0, 165, 255)
                        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 165, 255), 2)

                    # 3. Pose check
                    else:
                        pose_ok, pose_msg = face.check_pose(pose_dir)
                        if not pose_ok:
                            status, s_color = pose_msg, (255, 0, 255)
                            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (255, 0, 255), 2)

                        # 4. Quality check
                        else:
                            q_score, q_ok, q_reason = face.quality_check(frame)
                            if not q_ok:
                                status, s_color = f"Chat luong: {q_reason}", (0, 0, 255)
                                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 0, 255), 2)

                            # 5. Capture!
                            elif time.time() - last_t >= ENROLL_INTERVAL:
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
                    print("  [HUY]")
                    cv2.destroyAllWindows()
                    return

            all_embs.extend(step_embs)
            all_scores.extend(step_scores)
            print(f"  Buoc {step_i+1} xong: {instruction} ({len(step_embs)} anh)")

        cv2.destroyAllWindows()

        if len(all_embs) >= total_needed:
            # Select top-K best embeddings by quality + remove outliers
            from config import ENROLL_KEEP_TOP
            final_embs, final_scores = self.recognizer.select_best_embeddings(
                all_embs, all_scores, keep_top=ENROLL_KEEP_TOP
            )
            self.db.add_user(name, final_embs, final_scores)
            print(f"\n  [OK] DANG KY THANH CONG: '{name}'")
            print(f"       {len(all_embs)} captured -> {len(final_embs)} saved (top-{ENROLL_KEEP_TOP}) | FAISS: {self.db.total} vec")

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
                cleaned = self.recognizer.clean_embeddings(embs, OUTLIER_STD)
                self.db.add_user(user, cleaned)
                print(f"    -> {len(cleaned)} emb (loc tu {len(embs)})")

        print(f"  [DONE] FAISS: {self.db.total} vec | {len(self.db.get_users())} users")

    # ==================== VERIFY MODE ====================

    def verify_mode(self):
        users = self.db.get_users()
        if not users:
            print("\n  DB trong! Dang ky truoc.")
            return

        print(f"\n  === NHAN DIEN 24/7 | {len(users)} users | {self.db.total} vec ===")
        print(f"  TTA: {'ON' if TTA_ENABLED else 'OFF'} | 'r' ghi hinh | 'q' thoat")

        prev_t = 0
        recording, writer = False, None
        os.makedirs(RECORDS_DIR, exist_ok=True)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()
            fps = 1 / (now - prev_t) if prev_t > 0 else 0
            prev_t = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detector.detect(rgb)

            for face in faces:
                face.draw_mesh(frame)
                bx, by, bw, bh = face.bbox
                q_score, ok, reason = face.quality_check(frame)

                if not ok:
                    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (128,128,128), 1)
                    cv2.putText(frame, reason, (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128,128,128), 1)
                else:
                    emb = self.recognizer.get_embedding(frame, face.lm5)
                    if emb is not None:
                        name, score = self.db.match(emb)
                        accepted = score >= THRESHOLD
                        color = (0, 255, 0) if accepted else (0, 0, 255)
                        label = f"{name} ({score:.2f})" if accepted else f"Unknown ({score:.2f})"
                        status = "ACCEPT" if accepted else "REJECT"

                        self.draw_label(frame, label, face.bbox, color)
                        cv2.putText(frame, status, (bx, by+bh+18),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
                        # Quality indicator
                        cv2.putText(frame, f"Q:{q_score:.0%}", (bx+bw-45, by+bh+18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

            # HUD
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(frame, f"FAISS: {self.db.total} vec | 24/7 scanning",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

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
                    print(f"  [REC] Start")
                else:
                    recording = False
                    if writer:
                        writer.release()
                        writer = None
                    print("  [REC] Stop")

        if writer:
            writer.release()

    # ==================== MENU ====================

    def run(self):
        try:
            while True:
                u = self.db.get_users()
                print("\n" + "=" * 56)
                print(f"  MENU | FAISS: {self.db.total} vec | {len(u)} users")
                print("=" * 56)
                for n, c in u.items():
                    print(f"    - {n}: {c} emb")
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
            print("\n\n  [Ctrl+C] Dang tat...")

        self.cleanup()

    def _delete_user(self):
        users_list = list(self.db.get_users().items())
        if not users_list:
            print("  DB trong.")
            return

        print("\n  --- DANH SACH USER ---")
        for i, (n, c) in enumerate(users_list):
            print(f"    [{i+1}] {n}  ({c} emb)")
        print(f"    [0] Huy")

        try:
            sel = int(input(f"\n  Chon (0-{len(users_list)}): ").strip())
            if 1 <= sel <= len(users_list):
                name = users_list[sel - 1][0]
                if input(f"  Xoa '{name}'? (y/n): ").strip().lower() == 'y':
                    self.db.remove_user(name)
                    print(f"  [OK] Da xoa '{name}'.")
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
        print("[BYE]")


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
