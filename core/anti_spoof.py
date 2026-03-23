"""
AntiSpoofer v1.0 — Passive Liveness Detection (MiniFASNet ONNX).

Phát hiện ảnh giả mạo (Spoof) bằng cách phân tích vân ảnh (texture):
- Moiré pattern (nhiễu pixel màn hình điện thoại)
- Screen Glare (bóng phản quang màn hình)
- Paper texture (vân giấy in, thiếu chiều sâu 3D)

Model: best_model_quantized.onnx (~600KB) từ SuriAI/face-antispoof-onnx
Input : (batch, 3, 128, 128) — RGB, normalized [0,1]
Output: (batch, 2) — [real_logit, spoof_logit]
"""

import cv2
import numpy as np
import onnxruntime as ort
from logger import get_logger

logger = get_logger("antispoof")

# Giới hạn logit_diff để tránh overflow khi tính sigmoid
_LOGIT_CLIP = 50.0


class AntiSpoofer:
    """ONNX-based Passive Liveness Detection."""

    def __init__(self, model_path, img_size=128, threshold=0.5):
        """
        Args:
            model_path: Đường dẫn tới file .onnx anti-spoofing.
            img_size:   Kích thước input hình vuông (model yêu cầu 128x128).
            threshold:  Ngưỡng phân loại Real/Spoof (0.0-1.0).
                        Mặc định 0.5 (logit_diff >= 0 => Real).
        """
        self.model_path = model_path
        self.img_size = img_size
        self.threshold = threshold
        self.session = None

        try:
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"Loaded Anti-Spoofing model: {model_path} "
                        f"(input={self.input_name}, size={img_size})")
        except Exception as e:
            logger.error(f"Failed to load Anti-Spoofing model: {e}")

    # ------------------------------------------------------------------
    # PRIVATE: Crop & Preprocess
    # ------------------------------------------------------------------

    def _crop_face(self, img_bgr, bbox, expansion=1.5):
        """Cắt vùng mặt hình vuông với expansion padding.

        Args:
            img_bgr: Ảnh gốc BGR (HxWx3).
            bbox:    (x, y, w, h) — top-left + width/height.
            expansion: Hệ số mở rộng bounding box.
        Returns:
            Ảnh BGR đã crop + pad (square).
        """
        x, y, w, h = [int(v) for v in bbox]
        img_h, img_w = img_bgr.shape[:2]

        # Tính hình vuông mở rộng từ tâm mặt
        max_dim = max(w, h)
        cx, cy = x + w / 2, y + h / 2
        crop_size = int(max_dim * expansion)
        nx = int(cx - crop_size / 2)
        ny = int(cy - crop_size / 2)

        # Tính vùng cắt rộng hơn để thấy rõ viền điện thoại / mép ảnh in
        x1, y1 = max(0, nx), max(0, ny)
        x2, y2 = min(img_w, nx + crop_size), min(img_h, ny + crop_size)

        if x2 > x1 and y2 > y1:
            roi = img_bgr[y1:y2, x1:x2, :]
        else:
            return np.zeros((128, 128, 3), dtype=np.uint8)

        return roi

    def _preprocess(self, crop_bgr):
        """Squash resize → CHW tensor float32. Giống với PIL.Image.Resize trong lúc Train"""
        # BGR -> RGB giống lúc train
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize giống transforms.Resize (ép khung về 128x128 thay vì padding)
        resized = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # HWC → CHW → float32 [0, 1]
        tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        return tensor

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def is_real(self, img_bgr, bbox):
        """Kiểm tra khuôn mặt là người thật hay ảnh giả mạo.

        Args:
            img_bgr: Ảnh BGR gốc (HxWx3).
            bbox:    (x, y, w, h) bounding box khuôn mặt.

        Returns:
            (is_real: bool, liveness_score: float)
        """
        if self.session is None:
            return True, 1.0

        # Mở rộng vùng crop lên 2.2 để bao trọn mép điện thoại / viền ảnh in
        crop = self._crop_face(img_bgr, bbox, expansion=2.2)
        if crop.shape[0] < 2 or crop.shape[1] < 2:
            return False, 0.0

        # 2. Preprocess
        tensor = self._preprocess(crop)
        batch = np.expand_dims(tensor, axis=0)  # (1, 3, 128, 128)

        # 3. Inference
        try:
            logits = self.session.run(None, {self.input_name: batch})[0][0]
            real_logit = float(logits[0])
            spoof_logit = float(logits[1])
            logit_diff = real_logit - spoof_logit

            # Sigmoid: score = 1 / (1 + exp(-diff)), clip để tránh overflow
            clipped = np.clip(logit_diff, -_LOGIT_CLIP, _LOGIT_CLIP)
            liveness_score = float(1.0 / (1.0 + np.exp(-clipped)))

            # So sánh với threshold
            is_real = liveness_score >= self.threshold

            return is_real, liveness_score

        except Exception as e:
            logger.error(f"Anti-spoofing inference error: {e}")
            return False, 0.0
