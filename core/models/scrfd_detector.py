"""
SCRFDDetector v6.0 — InsightFace SCRFD wrapper.
Drop-in replacement for MediaPipe FaceDetector.
Returns the same FaceData objects, so service.py and recognizer.py work unchanged.

Usage:
    from models.scrfd_detector import SCRFDDetector
    detector = SCRFDDetector()
    faces = detector.detect(frame_rgb)
"""

import cv2
import numpy as np
import math
import os
from logger import get_logger

logger = get_logger("scrfd_detector")

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("insightface not installed. Run: pip install insightface onnxruntime")


class SCRFDFaceData:
    """Face data from SCRFD detection.
    
    Mimics the FaceData interface from MediaPipe detector so that
    service.py, recognizer.py, and all downstream code works unchanged.
    """

    def __init__(self, insightface_face, img_w, img_h):
        from config import (MIN_FACE_SIZE, BLUR_THRESH, MIN_BRIGHT, MAX_BRIGHT, MAX_TILT,
                          MIN_FACE_RATIO, MAX_FACE_RATIO,
                          GATING_EYE_OPENNESS_MIN, GATING_BBOX_JERK_MAX)
        
        self.img_w, self.img_h = img_w, img_h
        self._config_cache = {
            'MIN_FACE_SIZE': MIN_FACE_SIZE, 'BLUR_THRESH': BLUR_THRESH,
            'MIN_BRIGHT': MIN_BRIGHT, 'MAX_BRIGHT': MAX_BRIGHT,
            'MAX_TILT': MAX_TILT, 'MIN_FACE_RATIO': MIN_FACE_RATIO,
            'MAX_FACE_RATIO': MAX_FACE_RATIO,
            'GATING_EYE_OPENNESS_MIN': GATING_EYE_OPENNESS_MIN,
            'GATING_BBOX_JERK_MAX': GATING_BBOX_JERK_MAX,
        }
        
        # InsightFace face object
        self._raw = insightface_face
        
        # Bounding box: InsightFace gives [x1, y1, x2, y2]
        bbox_raw = insightface_face.bbox.astype(int)
        x1, y1, x2, y2 = bbox_raw
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        self.bbox = (x1, y1, x2 - x1, y2 - y1)  # (x, y, w, h)
        self.area = self.bbox[2] * self.bbox[3]
        
        # 5-point landmarks: InsightFace gives [[x,y], [x,y], ...]
        # Order: left_eye, right_eye, nose, left_mouth, right_mouth
        self.lm5 = insightface_face.kps.astype(np.float32)  # shape (5, 2)
        
        # Fake full 478 landmarks using 5 points (for compatibility)
        # Only indices used by config: LM_IDX_5PT = [33, 263, 1, 61, 291]
        self.lm2d = np.zeros((478, 2), dtype=np.float32)
        from config import LM_IDX_5PT
        for i, idx in enumerate(LM_IDX_5PT):
            self.lm2d[idx] = self.lm5[i]
        
        # Fill eye corner landmarks for eye_openness (approximate from kps)
        # Left eye corners: 33 (outer), 133 (inner) 
        self.lm2d[33] = self.lm5[0]   # left eye center
        self.lm2d[133] = self.lm5[0] + [10, 0]  # approximate inner corner
        # Right eye corners: 362 (inner), 263 (outer)
        self.lm2d[362] = self.lm5[1] - [10, 0]
        self.lm2d[263] = self.lm5[1]  # right eye center
        
        # Eye top/bottom for EAR (approximate)
        from config import LEFT_EYE_TOP, LEFT_EYE_BOTTOM, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM
        eye_h = max(self.bbox[3] * 0.04, 3)  # ~4% of face height
        self.lm2d[LEFT_EYE_TOP] = self.lm5[0] + [0, -eye_h]
        self.lm2d[LEFT_EYE_BOTTOM] = self.lm5[0] + [0, eye_h]
        self.lm2d[RIGHT_EYE_TOP] = self.lm5[1] + [0, -eye_h]
        self.lm2d[RIGHT_EYE_BOTTOM] = self.lm5[1] + [0, eye_h]
        
        # Nose landmark at index 1
        self.lm2d[1] = self.lm5[2]
        
        # BỔ SUNG QUAN TRỌNG: Ước lượng điểm cằm (Chin - index 152) để chạy 6DoF solvePnP
        # Do SCRFD không cung cấp điểm cằm -> Ngoại suy từ mũi và miệng
        # Khoảng cách từ miệng đến cằm ~ 0.8 -> 1.0 lần khoảng cách từ mũi đến miệng
        nose = self.lm5[2]
        mouth_center = (self.lm5[3] + self.lm5[4]) / 2.0
        vec_nose_to_mouth = mouth_center - nose
        chin = mouth_center + vec_nose_to_mouth * 0.85
        self.lm2d[152] = chin
        
        # Detection score from SCRFD
        self.det_score = float(insightface_face.det_score)
        
        # Tracking / gating attributes
        self.tracking_id = 0
        self.bbox_jerk = 0.0
        self.landmark_stability = 0.0
    
    def eye_openness(self):
        """Eye openness estimation (approximate from bbox ratio for SCRFD)."""
        # SCRFD doesn't have detailed eye landmarks, so we approximate
        # Based on face height ratio as proxy
        from config import LEFT_EYE_TOP, LEFT_EYE_BOTTOM, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM
        
        lt = self.lm2d[LEFT_EYE_TOP]
        lb = self.lm2d[LEFT_EYE_BOTTOM]
        l_vert = np.linalg.norm(lt - lb)
        l_horiz = max(np.linalg.norm(self.lm2d[33] - self.lm2d[133]), 1)
        left_ear = l_vert / l_horiz
        
        rt = self.lm2d[RIGHT_EYE_TOP]
        rb = self.lm2d[RIGHT_EYE_BOTTOM]
        r_vert = np.linalg.norm(rt - rb)
        r_horiz = max(np.linalg.norm(self.lm2d[362] - self.lm2d[263]), 1)
        right_ear = r_vert / r_horiz
        
        avg_ear = (left_ear + right_ear) / 2
        return float(left_ear), float(right_ear), float(avg_ear)

    def quality_check(self, frame):
        """Quality check — reuses same logic as MediaPipe FaceData."""
        cfg = self._config_cache
        x, y, w, h = self.bbox
        fh, fw = frame.shape[:2]

        if w < cfg['MIN_FACE_SIZE'] or h < cfg['MIN_FACE_SIZE']:
            return 0.0, False, "Nho"

        roi = frame[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]
        if roi.size == 0:
            return 0.0, False, "Rong"

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < cfg['BLUR_THRESH']:
            return 0.1, False, "Mo"

        bright = np.mean(gray)
        if bright < cfg['MIN_BRIGHT']:
            return 0.2, False, "Toi"
        if bright > cfg['MAX_BRIGHT']:
            return 0.2, False, "Sang"

        # Tilt angle from landmarks
        angle = abs(np.degrees(np.arctan2(
            self.lm5[1][1] - self.lm5[0][1],
            self.lm5[1][0] - self.lm5[0][0]
        )))
        if angle > cfg['MAX_TILT']:
            return 0.3, False, "Nghieng"

        if self.bbox_jerk > cfg['GATING_BBOX_JERK_MAX']:
            return 0.2, False, "Rung"

        # Composite score
        blur_score = min(blur / 200.0, 1.0)
        bright_score = 1.0 - abs(bright - 130) / 130.0
        angle_score = 1.0 - angle / 45.0
        face_size_score = min(w * h / (160 * 160), 1.0)
        det_score_bonus = min(self.det_score, 1.0) * 0.1  # SCRFD confidence bonus
        
        score = (blur_score * 0.35
                 + max(0, angle_score) * 0.20
                 + face_size_score * 0.20
                 + bright_score * 0.10
                 + det_score_bonus * 0.15)
        score = max(0.0, min(score, 1.0))
        return score, True, "OK"

    def distance_check(self, frame_w):
        """Check face distance from camera."""
        from config import MIN_FACE_RATIO, MAX_FACE_RATIO
        ratio = self.bbox[2] / frame_w
        if ratio < MIN_FACE_RATIO:
            return False, "QUA XA - Tien lai gan hon"
        if ratio > MAX_FACE_RATIO:
            return False, "QUA GAN - Lui ra xa hon"
        return True, "OK"

    def in_oval(self, frame_w, frame_h):
        """Check if face is inside guide oval."""
        cx, cy = frame_w / 2, frame_h / 2
        ax, ay = frame_w * 0.24, frame_h * 0.34
        nose = self.lm5[2]  # nose tip
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
        """Verify head matches required direction."""
        from config import (POSE_STRAIGHT_MAX, POSE_LEFT_MIN, POSE_RIGHT_MIN,
                          POSE_UP_MAX, POSE_DOWN_MIN)
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
        """Draw face landmarks on frame (simplified for SCRFD)."""
        # Draw bbox
        x, y, w, h = self.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Draw 5 keypoints
        for pt in self.lm5:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)


class SCRFDDetector:
    """SCRFD face detector from InsightFace.
    
    Drop-in replacement for MediaPipe FaceDetector.
    Returns list of SCRFDFaceData (same interface as FaceData).
    """

    def __init__(self, num_faces=3):
        from config import SCRFD_MODEL, SCRFD_CTX_ID, SCRFD_DET_SIZE, SCRFD_DET_THRESH
        
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("insightface not installed. Run: pip install insightface onnxruntime")
        
        self._app = FaceAnalysis(
            name=SCRFD_MODEL,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self._app.prepare(ctx_id=SCRFD_CTX_ID, det_size=SCRFD_DET_SIZE, det_thresh=SCRFD_DET_THRESH)
        self._max_faces = num_faces
        self._next_tracking_id = 0
        
        logger.info(f"SCRFD Detector initialized: model={SCRFD_MODEL}, ctx={SCRFD_CTX_ID}, "
                    f"det_size={SCRFD_DET_SIZE}, det_thresh={SCRFD_DET_THRESH}")

    def detect(self, frame_rgb, tracking_state=None):
        """Detect faces using SCRFD. Returns list of SCRFDFaceData.
        
        Args:
            frame_rgb: RGB frame (numpy array)
            tracking_state: optional dict for tracking across frames
        """
        # InsightFace expects BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        faces_raw = self._app.get(frame_bgr)
        
        if not faces_raw:
            return []
        
        h, w = frame_rgb.shape[:2]
        
        # Sort by area (largest first), limit to max_faces
        faces_raw = sorted(faces_raw, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        faces_raw = faces_raw[:self._max_faces]
        
        # Tracking state
        if tracking_state is None:
            ts_bboxes = {}
            ts_centroids = {}
        else:
            ts_bboxes = tracking_state.setdefault('bboxes', {})
            ts_centroids = tracking_state.setdefault('centroids', {})
        
        current_bboxes = {}
        current_centroids = {}
        used_tids = set()
        faces = []
        
        for raw_face in faces_raw:
            fd = SCRFDFaceData(raw_face, w, h)
            
            # Centroid tracking
            cx = fd.bbox[0] + fd.bbox[2] / 2
            cy = fd.bbox[1] + fd.bbox[3] / 2
            
            best_id = None
            min_dist = float('inf')
            
            for tid, (prev_cx, prev_cy) in ts_centroids.items():
                if tid in used_tids:
                    continue
                dist = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                tolerance = max(fd.bbox[2], fd.bbox[3]) * 0.5
                if dist < min_dist and dist < tolerance:
                    min_dist = dist
                    best_id = tid
            
            if best_id is None:
                best_id = self._next_tracking_id
                self._next_tracking_id += 1
            
            used_tids.add(best_id)
            fd.tracking_id = best_id
            current_centroids[best_id] = (cx, cy)
            current_bboxes[best_id] = fd.bbox
            
            # BBox jerk
            prev_bbox = ts_bboxes.get(best_id)
            if prev_bbox is not None:
                dx = abs(fd.bbox[0] - prev_bbox[0])
                dy = abs(fd.bbox[1] - prev_bbox[1])
                fd.bbox_jerk = math.sqrt(dx*dx + dy*dy)
            
            faces.append(fd)
        
        if tracking_state is not None:
            tracking_state['bboxes'] = current_bboxes
            tracking_state['centroids'] = current_centroids
        
        return faces

    def close(self):
        """Cleanup."""
        pass
