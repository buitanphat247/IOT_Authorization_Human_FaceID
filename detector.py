"""
FaceDetector - MediaPipe FaceLandmarker wrapper.
Handles face detection, landmark extraction, quality checks, head pose.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from config import *


class FaceDetector:
    """MediaPipe FaceLandmarker wrapper with quality & pose checks."""

    def __init__(self, mode="video", num_faces=3):
        if not os.path.exists(FL_PATH):
            import urllib.request, os as _os
            _os.makedirs(MODELS_DIR, exist_ok=True)
            print(f"  Downloading {os.path.basename(FL_PATH)}...")
            urllib.request.urlretrieve(FL_URL, FL_PATH)

        rm = vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=FL_PATH),
                running_mode=rm,
                num_faces=num_faces,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
            ))
        self.mode = mode
        self._ts = 0

    def detect(self, frame_rgb):
        """Detect faces. Returns list of FaceData or empty list."""
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if self.mode == "video":
            self._ts += 33
            res = self.landmarker.detect_for_video(mp_img, self._ts)
        else:
            res = self.landmarker.detect(mp_img)

        if not res.face_landmarks:
            return []

        h, w = frame_rgb.shape[:2]
        faces = []
        for fl in res.face_landmarks:
            fd = FaceData(fl, w, h)
            faces.append(fd)
        return faces

    def close(self):
        self.landmarker.close()


class FaceData:
    """Parsed face data from landmarks."""

    def __init__(self, landmarks, img_w, img_h):
        self.img_w, self.img_h = img_w, img_h
        # All 478 landmarks as pixel coords
        self.lm2d = np.array([[l.x * img_w, l.y * img_h] for l in landmarks], dtype=np.float32)
        # 5-point landmarks for ArcFace
        self.lm5 = self.lm2d[LM_IDX_5PT]
        # Bounding box
        pad = 15
        mins = self.lm2d.min(axis=0).astype(int) - pad
        maxs = self.lm2d.max(axis=0).astype(int) + pad
        self.bbox = (max(0, mins[0]), max(0, mins[1]), maxs[0] - mins[0], maxs[1] - mins[1])
        # Face size (area)
        self.area = self.bbox[2] * self.bbox[3]

    def quality_check(self, frame):
        """Full quality check. Returns (score 0-1, ok, reason)."""
        x, y, w, h = self.bbox
        fh, fw = frame.shape[:2]

        # Size check
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            return 0.0, False, "Nho"

        # ROI
        roi = frame[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]
        if roi.size == 0:
            return 0.0, False, "Rong"

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Blur check (Laplacian variance)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < BLUR_THRESH:
            return 0.1, False, "Mo"

        # Brightness
        bright = np.mean(gray)
        if bright < MIN_BRIGHT:
            return 0.2, False, "Toi"
        if bright > MAX_BRIGHT:
            return 0.2, False, "Sang"

        # Tilt angle
        angle = abs(np.degrees(np.arctan2(
            self.lm5[1][1] - self.lm5[0][1],
            self.lm5[1][0] - self.lm5[0][0]
        )))
        if angle > MAX_TILT:
            return 0.3, False, "Nghieng"

        # Composite quality score
        blur_score = min(blur / 200.0, 1.0)
        bright_score = 1.0 - abs(bright - 130) / 130.0
        angle_score = 1.0 - angle / 45.0
        score = blur_score * 0.4 + bright_score * 0.3 + max(0, angle_score) * 0.3
        return score, True, "OK"

    def distance_check(self, frame_w):
        """Check face distance from camera."""
        ratio = self.bbox[2] / frame_w
        if ratio < MIN_FACE_RATIO:
            return False, "QUA XA - Tien lai gan hon"
        if ratio > MAX_FACE_RATIO:
            return False, "QUA GAN - Lui ra xa hon"
        return True, "OK"

    def in_oval(self, frame_w, frame_h):
        """Check if face is inside guide oval."""
        cx, cy = frame_w // 2, frame_h // 2
        ax, ay = int(frame_w * 0.18), int(frame_h * 0.30)
        nose = self.lm2d[1]
        dx = (nose[0] - cx) / ax
        dy = (nose[1] - cy) / ay
        return (dx * dx + dy * dy) < 0.9

    def head_pose(self):
        """Get head pose offsets (h_offset, v_ratio)."""
        eye_cx = (self.lm5[0][0] + self.lm5[1][0]) / 2
        eye_cy = (self.lm5[0][1] + self.lm5[1][1]) / 2
        eye_dist = max(np.linalg.norm(self.lm5[1] - self.lm5[0]), 1)
        h_off = (self.lm5[2][0] - eye_cx) / eye_dist
        mouth_cy = (self.lm5[3][1] + self.lm5[4][1]) / 2
        face_h = max(mouth_cy - eye_cy, 1)
        v_ratio = (self.lm5[2][1] - eye_cy) / face_h
        return h_off, v_ratio

    def check_pose(self, direction):
        """Verify head matches required direction. Returns (ok, msg)."""
        h, v = self.head_pose()
        if direction == "straight":
            return (True, "OK") if abs(h) <= POSE_STRAIGHT_MAX else (False, "HAY NHIN THANG vao camera")
        elif direction == "left":
            return (True, "OK") if h <= POSE_LEFT_MIN else (False, "HAY QUAY DAU sang TRAI nhieu hon")
        elif direction == "right":
            return (True, "OK") if h >= POSE_RIGHT_MIN else (False, "HAY QUAY DAU sang PHAI nhieu hon")
        elif direction == "up":
            return (True, "OK") if v <= POSE_UP_MAX else (False, "HAY NGANG MAT len TREN")
        elif direction == "down":
            return (True, "OK") if v >= POSE_DOWN_MIN else (False, "HAY CUI MAT xuong DUOI")
        return True, "OK"

    def draw_mesh(self, frame):
        """Draw face mesh landmarks + contours on frame."""
        for pt in self.lm2d:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
        for contour in CONTOURS:
            for i in range(len(contour) - 1):
                p1 = (int(self.lm2d[contour[i]][0]), int(self.lm2d[contour[i]][1]))
                p2 = (int(self.lm2d[contour[i + 1]][0]), int(self.lm2d[contour[i + 1]][1]))
                cv2.line(frame, p1, p2, (0, 200, 0), 1)
