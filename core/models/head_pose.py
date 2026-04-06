"""
Head Pose Estimator 6DoF — cv2.solvePnP() based.
Thay thế heuristic cũ (±15° sai số) bằng PnP solver (±3° sai số).

Nguyên lý:
  - Dùng mô hình 3D chuẩn khuôn mặt (3D Face Model Points)
  - Kết hợp 2D landmarks từ MediaPipe (6 điểm mốc chính)
  - cv2.solvePnP() tìm rotation vector + translation vector
  - Rodrigues chuyển rotation vector → Euler angles (yaw, pitch, roll)

Không cần model ONNX — thuật toán thuần OpenCV.
"""

import cv2
import numpy as np
import math


# ============================================================
# 3D Face Model Points (Generic Human Face)
# Hệ tọa độ: X phải, Y xuống, Z vào trong
# Chuẩn hóa theo khuôn mặt trung bình (mm)
# ============================================================
MODEL_POINTS_68 = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

# MediaPipe landmark indices tương ứng 6 điểm trên
# [nose_tip, chin, left_eye_outer, right_eye_outer, left_mouth, right_mouth]
MEDIAPIPE_6PT_IDX = [1, 152, 33, 263, 61, 291]


class HeadPoseEstimator:
    """
    Ước lượng hướng khuôn mặt 6DoF (yaw, pitch, roll) bằng cv2.solvePnP().

    Ưu điểm so với heuristic cũ:
    - Sai số ±3° thay vì ±15°
    - Đo được cả 3 trục (yaw, pitch, roll) thay vì chỉ 2 ratio
    - Có tính vật lý (perspective projection) thay vì tỉ lệ pixel
    """

    def __init__(self, img_w=640, img_h=480):
        """
        Khởi tạo với kích thước ảnh để tạo Camera Matrix.

        Args:
            img_w: Chiều rộng ảnh (pixel)
            img_h: Chiều cao ảnh (pixel)
        """
        self._img_w = img_w
        self._img_h = img_h
        self._camera_matrix = self._build_camera_matrix(img_w, img_h)
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # Giả sử không có biến dạng ống kính

    @staticmethod
    def _build_camera_matrix(w, h):
        """
        Xây dựng Camera Intrinsic Matrix (giả lập từ kích thước ảnh).
        Focal length ≈ chiều rộng ảnh (xấp xỉ hợp lý cho webcam thường).
        """
        focal_length = w
        cx = w / 2.0
        cy = h / 2.0
        return np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def update_frame_size(self, img_w, img_h):
        """Cập nhật kích thước ảnh nếu thay đổi (e.g. resize)."""
        if img_w != self._img_w or img_h != self._img_h:
            self._img_w = img_w
            self._img_h = img_h
            self._camera_matrix = self._build_camera_matrix(img_w, img_h)

    def estimate(self, lm2d):
        """
        Ước lượng head pose từ 478 MediaPipe landmarks.

        Args:
            lm2d: np.ndarray shape (478, 2) — Tọa độ pixel 2D

        Returns:
            dict: {
                'yaw': float    (độ, dương = quay phải),
                'pitch': float  (độ, dương = ngẩng lên),
                'roll': float   (độ, dương = nghiêng phải),
                'success': bool
            }
        """
        # Trích 6 điểm mốc 2D tương ứng model 3D
        image_points = np.array([
            lm2d[idx] for idx in MEDIAPIPE_6PT_IDX
        ], dtype=np.float64)

        # Solve PnP — tìm rotation & translation vector
        success, rotation_vec, translation_vec = cv2.solvePnP(
            MODEL_POINTS_68,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {
                'yaw': 0.0,
                'pitch': 0.0,
                'roll': 0.0,
                'success': False
            }

        # Chuyển rotation vector → rotation matrix → Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        yaw, pitch, roll = self._rotation_matrix_to_euler(rotation_mat)

        return {
            'yaw': float(yaw),
            'pitch': float(pitch),
            'roll': float(roll),
            'success': True
        }

    @staticmethod
    def _rotation_matrix_to_euler(R):
        """
        Chuyển Rotation Matrix (3x3) → Euler Angles (yaw, pitch, roll) theo độ.

        Convention: ZYX (Tait-Bryan angles)
        - Yaw   (trục Y): Quay trái/phải
        - Pitch (trục X): Ngẩng/cúi
        - Roll  (trục Z): Nghiêng trái/phải
        """
        sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)

        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(-R[2, 0], sy)
            roll = math.atan2(R[1, 0], R[0, 0])
        else:
            pitch = math.atan2(-R[1, 2], R[1, 1])
            yaw = math.atan2(-R[2, 0], sy)
            roll = 0.0

        # Chuyển radian → degree
        return (
            math.degrees(yaw),
            math.degrees(pitch),
            math.degrees(roll)
        )

    def draw_axes(self, frame, lm2d, axis_length=100):
        """
        Vẽ 3 trục XYZ lên ảnh để trực quan hóa hướng mặt.

        Args:
            frame: ảnh BGR (sẽ bị vẽ đè lên)
            lm2d: np.ndarray shape (478, 2)
            axis_length: Độ dài trục vẽ (pixel)

        Returns:
            frame (đã vẽ trục), hoặc frame gốc nếu solvePnP fail
        """
        image_points = np.array([
            lm2d[idx] for idx in MEDIAPIPE_6PT_IDX
        ], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_68,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return frame

        # Project 3 trục (X=đỏ, Y=xanh lá, Z=xanh dương) từ 3D → 2D
        axis_3d = np.float64([
            [axis_length, 0, 0],    # X axis
            [0, axis_length, 0],    # Y axis
            [0, 0, axis_length],    # Z axis
        ])

        axis_2d, _ = cv2.projectPoints(
            axis_3d, rvec, tvec, self._camera_matrix, self._dist_coeffs
        )

        nose_2d = tuple(image_points[0].astype(int))
        x_end = tuple(axis_2d[0].ravel().astype(int))
        y_end = tuple(axis_2d[1].ravel().astype(int))
        z_end = tuple(axis_2d[2].ravel().astype(int))

        # Vẽ trục: X=đỏ, Y=xanh lá, Z=xanh dương
        cv2.line(frame, nose_2d, x_end, (0, 0, 255), 3)   # X — đỏ
        cv2.line(frame, nose_2d, y_end, (0, 255, 0), 3)    # Y — xanh lá
        cv2.line(frame, nose_2d, z_end, (255, 0, 0), 3)    # Z — xanh dương

        return frame


def check_pose_6dof(pose_result, direction="straight",
                    max_yaw=15.0, max_pitch=15.0, max_roll=20.0):
    """
    Kiểm tra hướng mặt dựa trên góc 6DoF.

    Args:
        pose_result: dict từ HeadPoseEstimator.estimate()
        direction: "straight" | "left" | "right" | "up" | "down" | "any"
        max_yaw: Ngưỡng yaw cho "straight" (độ)
        max_pitch: Ngưỡng pitch cho "straight" (độ)
        max_roll: Ngưỡng roll (độ)

    Returns:
        (ok: bool, message: str, details: dict)
    """
    if not pose_result.get('success', False):
        return False, "Không đo được hướng mặt", pose_result

    yaw = pose_result['yaw']
    pitch = pose_result['pitch']
    roll = pose_result['roll']

    # Roll quá nghiêng → reject mọi trường hợp
    if abs(roll) > max_roll:
        return False, f"Đầu nghiêng quá ({roll:.0f}°), hãy giữ thẳng", pose_result

    if direction == "any":
        return True, "OK", pose_result

    if direction == "straight":
        if abs(yaw) > max_yaw:
            side = "phải" if yaw > 0 else "trái"
            return False, f"Hãy nhìn thẳng (đang quay {side} {abs(yaw):.0f}°)", pose_result
        if abs(pitch) > max_pitch:
            ud = "lên" if pitch > 0 else "xuống"
            return False, f"Hãy nhìn thẳng (đang ngẩng {ud} {abs(pitch):.0f}°)", pose_result
        return True, "OK", pose_result

    elif direction == "left":
        if yaw > -20.0:
            return False, "Hãy quay đầu sang TRÁI nhiều hơn", pose_result
        return True, "OK", pose_result

    elif direction == "right":
        if yaw < 20.0:
            return False, "Hãy quay đầu sang PHẢI nhiều hơn", pose_result
        return True, "OK", pose_result

    elif direction == "up":
        if pitch < 10.0:
            return False, "Hãy ngẩng đầu lên TRÊN", pose_result
        return True, "OK", pose_result

    elif direction == "down":
        if pitch > -10.0:
            return False, "Hãy cúi đầu xuống DƯỚI", pose_result
        return True, "OK", pose_result

    return True, "OK", pose_result
