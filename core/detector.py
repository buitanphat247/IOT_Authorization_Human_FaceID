"""
FaceDetector v5.0 - MediaPipe FaceLandmarker wrapper.
Enhanced: Eye openness check, BBox jerk detection, Landmark stability gating.
"""

import cv2
import numpy as np
import math
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from config import (FL_PATH, FL_URL, MODELS_DIR, LM_IDX_5PT,
                    MIN_FACE_SIZE, BLUR_THRESH, MIN_BRIGHT, MAX_BRIGHT, MAX_TILT,
                    MIN_FACE_RATIO, MAX_FACE_RATIO,
                    GATING_EYE_OPENNESS_MIN, GATING_BBOX_JERK_MAX, GATING_LANDMARK_STABILITY,
                    LEFT_EYE_TOP, LEFT_EYE_BOTTOM, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                    POSE_STRAIGHT_MAX, POSE_LEFT_MIN, POSE_RIGHT_MIN, POSE_UP_MAX, POSE_DOWN_MIN,
                    FACE_OVAL, LEFT_EYE, RIGHT_EYE, LIPS, L_BROW, R_BROW, NOSE_BRIDGE, CONTOURS)
import os
from logger import get_logger

logger = get_logger("detector")

def get_center_face(faces, frame_w, frame_h):
    """
    Tìm khuôn mặt nằm gần tâm khung hình nhất.
    Bỏ qua tất cả các khuôn mặt khác xung quanh.
    """
    if not faces:
        return None
        
    center_x = frame_w / 2
    center_y = frame_h / 2
    
    closest_face = None
    min_distance = float('inf')
    
    for face in faces:
        nose_x, nose_y = face.lm2d[1]
        distance = math.sqrt((nose_x - center_x)**2 + (nose_y - center_y)**2)
        if distance < min_distance:
            min_distance = distance
            closest_face = face
            
    return closest_face

class FaceDetector:
    """MediaPipe FaceLandmarker wrapper with quality & pose checks."""

    def __init__(self, mode="video", num_faces=3):
        if not os.path.exists(FL_PATH):
            import urllib.request, os as _os
            _os.makedirs(MODELS_DIR, exist_ok=True)
            logger.info(f"Downloading {os.path.basename(FL_PATH)}...")
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
        # Lưu trạng thái frame trước để tính Jerk / Stability
        self._next_tracking_id = 0

    def detect(self, frame_rgb, tracking_state=None):
        """Detect faces. Returns list of FaceData or empty list.
        Provide tracking_state dict to maintain tracking across contiguous frames.
        """
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        if self.mode == "video":
            self._ts = int(time.time() * 1000)
            res = self.landmarker.detect_for_video(mp_img, self._ts)
        else:
            res = self.landmarker.detect(mp_img)

        if not res.face_landmarks:
            return []

        h, w = frame_rgb.shape[:2]
        faces = []
        
        # State isolation per request
        if tracking_state is None:
            # Stateless if not provided
            ts_bboxes = {}
            ts_landmarks = {}
            ts_centroids = {}
        else:
            ts_bboxes = tracking_state.setdefault('bboxes', {})
            ts_landmarks = tracking_state.setdefault('landmarks', {})
            ts_centroids = tracking_state.setdefault('centroids', {})
            
        current_bboxes = {}
        current_landmarks = {}
        current_centroids = {}
        used_tids = set()
        
        for fl in res.face_landmarks:
            fd = FaceData(fl, w, h)
            
            # --- Centroid Tracking logic (Điểm 7) ---
            cx = fd.bbox[0] + fd.bbox[2] / 2
            cy = fd.bbox[1] + fd.bbox[3] / 2
            
            best_id = None
            min_dist = float('inf')
            
            # Tìm khuôn mặt gần nhất ở frame trước
            for tid, (prev_cx, prev_cy) in ts_centroids.items():
                if tid in used_tids:
                    continue
                dist = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                # Ngưỡng tracking hợp lệ: không đi chuyển quá nửa kích thước khuôn mặt
                tolerance = max(fd.bbox[2], fd.bbox[3]) * 0.5
                if dist < min_dist and dist < tolerance:
                    min_dist = dist
                    best_id = tid
                    
            if best_id is None:
                # Khuôn mặt mới xuất hiện
                best_id = self._next_tracking_id
                self._next_tracking_id += 1
                
            used_tids.add(best_id)
            fd.tracking_id = best_id
            current_centroids[best_id] = (cx, cy)
            current_bboxes[best_id] = fd.bbox
            current_landmarks[best_id] = fd.lm2d
            
            # Tính BBox Jerk (Kiểm tra xem mặt có rung không)
            prev_bbox = ts_bboxes.get(best_id)
            if prev_bbox is not None:
                dx = abs(fd.bbox[0] - prev_bbox[0])
                dy = abs(fd.bbox[1] - prev_bbox[1])
                fd.bbox_jerk = math.sqrt(dx*dx + dy*dy)
            else:
                fd.bbox_jerk = 0.0
            
            # Tính Landmark Stability
            prev_lm = ts_landmarks.get(best_id)
            if prev_lm is not None and GATING_LANDMARK_STABILITY:
                # Trung bình khoảng cách di chuyển của 5 điểm mốc chính
                diffs = np.linalg.norm(fd.lm5 - prev_lm[LM_IDX_5PT], axis=1)
                fd.landmark_stability = float(np.mean(diffs))
            else:
                fd.landmark_stability = 0.0
            
            faces.append(fd)
            
        # Lưu lại state cho frame kế nếu có truyền tracking_state
        if tracking_state is not None:
            tracking_state['bboxes'] = current_bboxes
            tracking_state['landmarks'] = current_landmarks
            tracking_state['centroids'] = current_centroids
            
        return faces

    def close(self):
        self.landmarker.close()


class FaceData:
    """Parsed face data from landmarks with enhanced quality checks."""

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
        # Enhanced gating attributes (set by FaceDetector)
        self.tracking_id = 0
        self.bbox_jerk = 0.0
        self.landmark_stability = 0.0

    def eye_openness(self):
        """Tính tỷ lệ mở mắt (Eye Aspect Ratio heuristic).
        Dùng khoảng cách dọc mí trên-dưới / khoảng cách ngang hai đuôi mắt.
        Returns (left_ear, right_ear, avg_ear).
        """
        # Mắt trái
        lt = self.lm2d[LEFT_EYE_TOP]
        lb = self.lm2d[LEFT_EYE_BOTTOM]
        l_vert = np.linalg.norm(lt - lb)
        l_horiz = max(np.linalg.norm(self.lm2d[33] - self.lm2d[133]), 1)  # đuôi mắt trái
        left_ear = l_vert / l_horiz
        
        # Mắt phải
        rt = self.lm2d[RIGHT_EYE_TOP]
        rb = self.lm2d[RIGHT_EYE_BOTTOM]
        r_vert = np.linalg.norm(rt - rb)
        r_horiz = max(np.linalg.norm(self.lm2d[362] - self.lm2d[263]), 1)  # đuôi mắt phải
        right_ear = r_vert / r_horiz
        
        avg_ear = (left_ear + right_ear) / 2
        return float(left_ear), float(right_ear), float(avg_ear)

    def quality_check(self, frame):
        """Full quality check with enhanced gating.
        Returns (score 0-1, ok, reason).
        """
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

        # Brightness (BUG-06 fix: thêm local-contrast analysis)
        bright = np.mean(gray)
        if bright < MIN_BRIGHT:
            return 0.2, False, "Toi"
        if bright > MAX_BRIGHT:
            return 0.2, False, "Sang"

        # Local Contrast: chia ROI thành 4 phần, đo chênh lệch sáng
        roi_h, roi_w = gray.shape[:2]
        if roi_h > 10 and roi_w > 10:
            h_mid, w_mid = roi_h // 2, roi_w // 2
            quadrants = [gray[:h_mid, :w_mid], gray[:h_mid, w_mid:],
                         gray[h_mid:, :w_mid], gray[h_mid:, w_mid:]]
            brightnesses = [float(np.mean(q)) for q in quadrants]
            contrast_ratio = max(brightnesses) / max(min(brightnesses), 1)
            if contrast_ratio > 2.5:
                return 0.2, False, "NgSang"

        # Tilt angle
        angle = abs(np.degrees(np.arctan2(
            self.lm5[1][1] - self.lm5[0][1],
            self.lm5[1][0] - self.lm5[0][0]
        )))
        if angle > MAX_TILT:
            return 0.3, False, "Nghieng"

        # --- ENHANCED GATING (Điểm 7) ---
        # Eye openness check
        _, _, avg_ear = self.eye_openness()
        if avg_ear < GATING_EYE_OPENNESS_MIN:
            return 0.15, False, "NhamMat"
        
        # BBox jerk check (mặt rung quá mạnh)
        if self.bbox_jerk > GATING_BBOX_JERK_MAX:
            return 0.2, False, "Rung"

        # Composite quality score — Data-driven weights (tối ưu từ phân tích)
        # Blur là predictor mạnh nhất của chất lượng embedding → tăng lên 40%
        # Size ảnh hưởng lớn đến chi tiết khuôn mặt → tăng lên 20%
        # Angle ảnh hưởng đến alignment → giữ 20%
        # Brightness và Eye phụ thuộc nhau → giảm xuống
        blur_score = min(blur / 200.0, 1.0)
        bright_score = 1.0 - abs(bright - 130) / 130.0
        angle_score = 1.0 - angle / 45.0
        eye_score = min(avg_ear / 0.05, 1.0)  # normalize EAR
        face_size_score = min(w * h / (160 * 160), 1.0)  # lớn hơn → tốt hơn
        
        # Landmark stability penalty: mặt đang di chuyển nhanh → giảm quality
        stability_penalty = 0.0
        if self.landmark_stability > 5.0:
            stability_penalty = min(self.landmark_stability / 20.0, 0.15)
        
        score = (blur_score * 0.40
                 + max(0, angle_score) * 0.20
                 + face_size_score * 0.20
                 + bright_score * 0.10
                 + eye_score * 0.10
                 - stability_penalty)
        score = max(0.0, min(score, 1.0))
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
        """Check if face is inside guide oval (matches CSS: 48% width, 68% height)."""
        cx, cy = frame_w / 2, frame_h / 2
        ax, ay = frame_w * 0.24, frame_h * 0.34  # half-axes matching CSS overlay
        nose = self.lm2d[1]
        dx = (nose[0] - cx) / ax
        dy = (nose[1] - cy) / ay
        return (dx * dx + dy * dy) < 0.85

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
        if direction == "any":
            return True, "OK"
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
