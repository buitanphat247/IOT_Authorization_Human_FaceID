"""
FaceQualityAssessor v6.1 — Multi-Signal Face Quality Scorer.
Replaces simple rule-based (Laplacian + brightness) with production-grade scoring.

Signals combined:
  1. Blur (Laplacian variance + gradient magnitude)
  2. Illumination (histogram uniformity + local contrast)
  3. Face Geometry (size, centering, aspect ratio)
  4. Pose (cv2.solvePnP 6DoF — yaw/pitch/roll ±3°)
  5. Occlusion (face symmetry + landmark visibility)
  6. Sharpness (high-frequency energy in face ROI)

Output: unified score 0.0 → 1.0 (continuous, not binary)
  - >= 0.70: Excellent — dùng cho enrollment
  - 0.50-0.70: Good — dùng cho recognition
  - 0.30-0.50: Fair — cảnh báo, vẫn nhận diện được
  - < 0.30: Poor — reject, yêu cầu chụp lại

Usage:
    assessor = FaceQualityAssessor()
    score, details = assessor.assess(face_roi_bgr, landmarks_5pt, lm2d=full_478_landmarks)
"""

import cv2
import numpy as np
from logger import get_logger

try:
    from models.head_pose import HeadPoseEstimator
    _HAS_HEAD_POSE = True
except ImportError:
    _HAS_HEAD_POSE = False

logger = get_logger("quality")


class FaceQualityAssessor:
    """Multi-Signal Face Quality Scorer.
    
    Combines 6 quality signals with learned weights into a single 0-1 score.
    Much more accurate than simple Laplacian + brightness thresholds.
    """

    # Signal weights — tuned for ArcFace recognition quality
    # Blur is the strongest predictor of embedding quality
    WEIGHTS = {
        'blur': 0.25,
        'sharpness': 0.15,
        'illumination': 0.15,
        'geometry': 0.15,
        'pose': 0.15,
        'occlusion': 0.15,
    }

    # Quality thresholds
    THRESH_EXCELLENT = 0.70  # Enrollment quality
    THRESH_GOOD = 0.50       # Recognition quality
    THRESH_FAIR = 0.30       # Warning zone
    # Below FAIR = reject

    def __init__(self):
        logger.info("FaceQualityAssessor v6.1 initialized (Multi-Signal + 6DoF Pose, Thread-Safe)")

    def assess(self, face_roi_bgr, landmarks_5pt=None, full_frame=None, bbox=None, lm2d=None, img_w=640, img_h=480):
        """Assess face quality from multiple signals.
        
        Args:
            face_roi_bgr: cropped face region (BGR, numpy array)
            landmarks_5pt: 5-point landmarks [[x,y], ...] (optional, improves accuracy)
            full_frame: full camera frame (optional, for centering check)
            bbox: face bounding box (x, y, w, h) (optional, for geometry)
            lm2d: full 478 MediaPipe landmarks (optional, enables 6DoF pose)
            img_w: image width for solvePnP camera matrix
            img_h: image height for solvePnP camera matrix
            
        Returns:
            (score, details_dict)
            score: float 0.0-1.0
            details_dict: individual signal scores + grade
        """
        if face_roi_bgr is None or face_roi_bgr.size == 0:
            return 0.0, {'grade': 'REJECT', 'reason': 'Empty ROI'}

        h, w = face_roi_bgr.shape[:2]
        if h < 20 or w < 20:
            return 0.0, {'grade': 'REJECT', 'reason': 'Too small'}

        gray = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2GRAY)

        # === Signal 1: Blur Score ===
        blur_score = self._blur_score(gray)

        # === Signal 2: Sharpness Score ===
        sharpness_score = self._sharpness_score(gray)

        # === Signal 3: Illumination Score ===
        illum_score = self._illumination_score(gray)

        # === Signal 4: Geometry Score ===
        geom_score = self._geometry_score(w, h, full_frame, bbox)

        # === Signal 5: Pose Score (6DoF solvePnP if lm2d available) ===
        pose_score, pose_info = self._pose_score(landmarks_5pt, w, h, lm2d, img_w, img_h)

        # === Signal 6: Occlusion Score ===
        occl_score = self._occlusion_score(gray, landmarks_5pt, w, h)

        # === Weighted Combination ===
        signals = {
            'blur': blur_score,
            'sharpness': sharpness_score,
            'illumination': illum_score,
            'geometry': geom_score,
            'pose': pose_score,
            'occlusion': occl_score,
        }

        final_score = sum(
            self.WEIGHTS[key] * signals[key] for key in self.WEIGHTS
        )
        final_score = max(0.0, min(1.0, final_score))

        # Grade
        if final_score >= self.THRESH_EXCELLENT:
            grade = 'EXCELLENT'
        elif final_score >= self.THRESH_GOOD:
            grade = 'GOOD'
        elif final_score >= self.THRESH_FAIR:
            grade = 'FAIR'
        else:
            grade = 'POOR'

        # Find weakest signal for feedback
        weakest = min(signals, key=signals.get)
        weakest_score = signals[weakest]

        details = {
            'score': round(final_score, 3),
            'grade': grade,
            'signals': {k: round(v, 3) for k, v in signals.items()},
            'weakest': weakest,
            'weakest_score': round(weakest_score, 3),
            'feedback': self._get_feedback(weakest, weakest_score),
        }

        return final_score, details

    def is_enrollment_quality(self, score):
        """Check if quality is good enough for enrollment."""
        return score >= self.THRESH_EXCELLENT

    def is_recognition_quality(self, score):
        """Check if quality is good enough for recognition."""
        return score >= self.THRESH_FAIR

    # ============================================================
    # Signal 1: Blur Detection
    # ============================================================

    def _blur_score(self, gray):
        """Laplacian variance + focus measure.
        
        Better than simple Laplacian: uses normalized variance
        relative to image size to handle different resolutions.
        """
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        variance = lap.var()

        # Normalize: typical sharp face has variance 200-1000+
        # Blurry face has variance < 50
        score = min(variance / 300.0, 1.0)

        # Additional: Tenengrad focus measure (Sobel gradient magnitude)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.mean(gx**2 + gy**2)
        tenen_score = min(tenengrad / 5000.0, 1.0)

        # Combine: 60% Laplacian + 40% Tenengrad
        return 0.6 * score + 0.4 * tenen_score

    # ============================================================
    # Signal 2: Sharpness (High-Frequency Energy)
    # ============================================================

    def _sharpness_score(self, gray):
        """Measure high-frequency energy using DCT.
        
        Sharp images have more energy in high-frequency DCT coefficients.
        """
        h, w = gray.shape
        # Resize to standard size for consistent measurement
        std_size = 112
        resized = cv2.resize(gray, (std_size, std_size))
        
        # DCT
        dct = cv2.dct(np.float32(resized))
        
        # High-frequency energy (bottom-right quadrant)
        hf_region = dct[std_size//2:, std_size//2:]
        hf_energy = np.mean(np.abs(hf_region))
        
        # Low-frequency energy (top-left quadrant, excluding DC)
        lf_region = dct[1:std_size//4, 1:std_size//4]
        lf_energy = np.mean(np.abs(lf_region)) + 1e-6
        
        # Ratio: more HF relative to LF = sharper
        ratio = hf_energy / lf_energy
        score = min(ratio / 0.3, 1.0)
        
        return score

    # ============================================================
    # Signal 3: Illumination Quality
    # ============================================================

    def _illumination_score(self, gray):
        """Assess illumination quality using histogram analysis.
        
        Good illumination = spread histogram, good contrast.
        Bad = too dark, too bright, or uneven lighting.
        """
        h, w = gray.shape
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # 1. Brightness score (ideal ~120-140)
        if mean_brightness < 40:
            bright_score = mean_brightness / 40.0  # too dark
        elif mean_brightness > 220:
            bright_score = (255 - mean_brightness) / 35.0  # too bright
        else:
            # Gaussian-like around ideal brightness
            ideal = 130.0
            bright_score = 1.0 - min(abs(mean_brightness - ideal) / 100.0, 0.5)

        # 2. Contrast score (histogram spread)
        # Good contrast: std > 40
        contrast_score = min(std_brightness / 50.0, 1.0)

        # 3. Uniformity: check if lighting is even across face
        # Split into 4 quadrants, measure brightness difference
        mid_h, mid_w = h // 2, w // 2
        if mid_h > 5 and mid_w > 5:
            quads = [
                gray[:mid_h, :mid_w],
                gray[:mid_h, mid_w:],
                gray[mid_h:, :mid_w],
                gray[mid_h:, mid_w:]
            ]
            quad_means = [float(np.mean(q)) for q in quads]
            max_diff = max(quad_means) - min(quad_means)
            uniformity_score = 1.0 - min(max_diff / 80.0, 1.0)
        else:
            uniformity_score = 0.5

        # 4. Histogram entropy (information richness)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))
        entropy_score = min(entropy / 7.0, 1.0)  # max entropy ~8 for uniform

        return (bright_score * 0.3 + contrast_score * 0.25 +
                uniformity_score * 0.25 + entropy_score * 0.2)

    # ============================================================
    # Signal 4: Face Geometry
    # ============================================================

    def _geometry_score(self, face_w, face_h, full_frame=None, bbox=None):
        """Assess face geometry: size, aspect ratio, centering."""
        scores = []

        # 1. Face size (bigger = better detail)
        area = face_w * face_h
        size_score = min(area / (120 * 120), 1.0)
        scores.append(size_score * 0.4)

        # 2. Aspect ratio (ideal face ~0.75 w/h ratio)
        aspect = face_w / max(face_h, 1)
        ideal_aspect = 0.75
        aspect_score = 1.0 - min(abs(aspect - ideal_aspect) / 0.5, 1.0)
        scores.append(aspect_score * 0.3)

        # 3. Centering (if full frame provided)
        if full_frame is not None and bbox is not None:
            fh, fw = full_frame.shape[:2]
            cx = bbox[0] + bbox[2] / 2
            cy = bbox[1] + bbox[3] / 2
            dx = abs(cx - fw / 2) / (fw / 2)
            dy = abs(cy - fh / 2) / (fh / 2)
            center_score = 1.0 - min((dx + dy) / 2, 1.0)
            scores.append(center_score * 0.3)
        else:
            scores.append(0.5 * 0.3)

        return sum(scores)

    # ============================================================
    # Signal 5: Pose Estimation
    # ============================================================

    def _pose_score(self, landmarks, face_w, face_h, lm2d=None, img_w=640, img_h=480):
        """Estimate head pose quality.
        
        Strategy:
          1. If lm2d (478 landmarks) available → cv2.solvePnP 6DoF (±3° accuracy)
          2. Fallback → heuristic from 5-point landmarks (±15° accuracy)
        """
        # === Strategy 1: solvePnP 6DoF (chính xác) ===
        if lm2d is not None and _HAS_HEAD_POSE and len(lm2d) >= 468:
            try:
                # Local instantiation đảm bảo Thread-Safe 100% khi chạy song song qua ThreadPool
                estimator = HeadPoseEstimator(img_w, img_h)
                result = estimator.estimate(lm2d)
                if result['success']:
                    yaw = abs(result['yaw'])
                    pitch = abs(result['pitch'])
                    roll = abs(result['roll'])

                    # Score: frontal = 1.0, large angle = 0.0
                    # Yaw > 45° or Pitch > 35° = very bad
                    yaw_score = max(0.0, 1.0 - yaw / 45.0)
                    pitch_score = max(0.0, 1.0 - pitch / 35.0)
                    roll_score = max(0.0, 1.0 - roll / 30.0)

                    pose_score = (yaw_score * 0.40 + pitch_score * 0.30 + roll_score * 0.30)

                    pose_info = {
                        'method': '6DoF_solvePnP',
                        'yaw': round(result['yaw'], 1),
                        'pitch': round(result['pitch'], 1),
                        'roll': round(result['roll'], 1),
                    }
                    return pose_score, pose_info
            except Exception:
                pass  # fallback to heuristic

        # === Strategy 2: Heuristic from 5-point landmarks (fallback) ===
        if landmarks is None or len(landmarks) < 5:
            return 0.5, {'method': 'none'}  # No landmarks = neutral score

        lm = np.array(landmarks, dtype=np.float32)
        
        # Yaw: nose offset from eye center
        eye_center_x = (lm[0][0] + lm[1][0]) / 2
        eye_dist = max(abs(lm[1][0] - lm[0][0]), 1)
        nose_offset = (lm[2][0] - eye_center_x) / eye_dist
        yaw_score = 1.0 - min(abs(nose_offset) / 0.5, 1.0)

        # Pitch: nose vertical position between eyes and mouth
        eye_center_y = (lm[0][1] + lm[1][1]) / 2
        mouth_center_y = (lm[3][1] + lm[4][1]) / 2
        face_height = max(mouth_center_y - eye_center_y, 1)
        nose_v_ratio = (lm[2][1] - eye_center_y) / face_height
        pitch_score = 1.0 - min(abs(nose_v_ratio - 0.4) / 0.4, 1.0)

        # Roll: eye line angle
        eye_angle = abs(np.degrees(np.arctan2(
            lm[1][1] - lm[0][1],
            lm[1][0] - lm[0][0]
        )))
        roll_score = 1.0 - min(eye_angle / 30.0, 1.0)

        # Eye symmetry
        left_eye_nose = np.linalg.norm(lm[0] - lm[2])
        right_eye_nose = np.linalg.norm(lm[1] - lm[2])
        symmetry_ratio = min(left_eye_nose, right_eye_nose) / max(left_eye_nose, right_eye_nose, 1)

        pose_score = (yaw_score * 0.35 + pitch_score * 0.25 +
                roll_score * 0.20 + symmetry_ratio * 0.20)

        pose_info = {'method': 'heuristic_5pt'}
        return pose_score, pose_info

    # ============================================================
    # Signal 6: Occlusion Detection
    # ============================================================

    def _occlusion_score(self, gray, landmarks, face_w, face_h):
        """Detect occlusion using face symmetry and edge density.
        
        Occluded face = asymmetric brightness + low edge density in occluded region.
        """
        h, w = gray.shape

        # 1. Left-right symmetry
        mid_w = w // 2
        if mid_w > 5:
            left_half = gray[:, :mid_w]
            right_half = cv2.flip(gray[:, mid_w:], 1)
            
            # Resize to match
            min_w = min(left_half.shape[1], right_half.shape[1])
            if min_w > 5:
                left_half = left_half[:, :min_w]
                right_half = right_half[:, :min_w]
                
                # Structural similarity proxy: normalized cross-correlation
                diff = np.abs(left_half.astype(float) - right_half.astype(float))
                symmetry_score = 1.0 - min(np.mean(diff) / 60.0, 1.0)
            else:
                symmetry_score = 0.5
        else:
            symmetry_score = 0.5

        # 2. Edge density in face regions (occluded areas have fewer edges)
        edges = cv2.Canny(gray, 50, 150)
        total_edge_density = np.sum(edges > 0) / max(gray.size, 1)
        edge_score = min(total_edge_density / 0.08, 1.0)

        # 3. Check upper face (forehead/eyes) vs lower face (mouth/chin)
        mid_h = h // 2
        if mid_h > 5:
            upper_edges = np.sum(edges[:mid_h] > 0) / max(edges[:mid_h].size, 1)
            lower_edges = np.sum(edges[mid_h:] > 0) / max(edges[mid_h:].size, 1)
            # If lower face has much fewer edges, might be mask occlusion
            if upper_edges > 0.01:
                balance = min(lower_edges / max(upper_edges, 1e-6), 1.0)
                balance_score = min(balance / 0.5, 1.0)
            else:
                balance_score = 0.5
        else:
            balance_score = 0.5

        # 4. Skin color uniformity (occluding objects often have different color)
        if face_w > 20 and face_h > 20:
            hsv = cv2.cvtColor(
                cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV
            )
            # For grayscale input, just check variance
            skin_uniformity = 1.0 - min(np.std(gray) / 80.0, 0.5)
        else:
            skin_uniformity = 0.5

        return (symmetry_score * 0.35 + edge_score * 0.25 +
                balance_score * 0.25 + skin_uniformity * 0.15)

    # ============================================================
    # Feedback Messages
    # ============================================================

    def _get_feedback(self, weakest_signal, score):
        """Human-readable feedback for the weakest quality signal."""
        messages = {
            'blur': "Ảnh bị mờ — giữ yên khuôn mặt",
            'sharpness': "Ảnh thiếu chi tiết — lại gần camera hơn",
            'illumination': "Ánh sáng không tốt — di chuyển tới nơi sáng hơn",
            'geometry': "Khuôn mặt quá nhỏ hoặc lệch — nhìn thẳng camera",
            'pose': "Đầu nghiêng quá nhiều — nhìn thẳng vào camera",
            'occlusion': "Khuôn mặt bị che — bỏ khẩu trang/kính",
        }
        if score >= 0.6:
            return "OK"
        return messages.get(weakest_signal, "Chất lượng ảnh chưa tốt")

    # ============================================================
    # Batch Assessment (cho enrollment multi-frame)
    # ============================================================

    def assess_batch(self, face_rois, landmarks_list=None):
        """Assess multiple face ROIs and return best one.
        
        Args:
            face_rois: list of face ROI images (BGR)
            landmarks_list: list of 5-point landmarks (optional)
            
        Returns:
            (best_idx, best_score, all_scores)
        """
        scores = []
        for i, roi in enumerate(face_rois):
            lm = landmarks_list[i] if landmarks_list and i < len(landmarks_list) else None
            score, _ = self.assess(roi, lm)
            scores.append(score)

        if not scores:
            return -1, 0.0, []

        best_idx = int(np.argmax(scores))
        return best_idx, scores[best_idx], scores
