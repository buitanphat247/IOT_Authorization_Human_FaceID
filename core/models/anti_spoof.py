import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import onnxruntime as ort
from logger import get_logger

logger = get_logger("antispoof")

# Giới hạn logit_diff để tránh overflow khi tính sigmoid
_LOGIT_CLIP = 50.0


class AntiSpoofer:
    """Hybrid Passive Liveness Detection (Supports ONNX and PyTorch)."""

    def __init__(self, model_path, img_size=128, threshold=0.5):
        self.model_path = model_path
        self.img_size = img_size
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.is_onnx = model_path.endswith(".onnx")
        self.session = None
        self.torch_model = None

        try:
            if self.is_onnx:
                self.session = ort.InferenceSession(
                    model_path,
                    providers=["CPUExecutionProvider"]
                )
                self.input_name = self.session.get_inputs()[0].name
                logger.info(f"Loaded ONNX Anti-Spoofing model: {model_path} (size={img_size})")
            else:
                from .minifas_v2 import get_minifasnet_v2
                self.torch_model = get_minifasnet_v2(num_classes=3)
                state_dict = torch.load(model_path, map_location=self.device)
                
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                
                self.torch_model.load_state_dict(new_state_dict)
                self.torch_model.to(self.device)
                self.torch_model.eval()
                
                if "80x80" in model_path:
                    self.img_size = 80
                
                logger.info(f"Loaded PyTorch Anti-Spoofing model: {model_path} (size={self.img_size})")
        except Exception as e:
            logger.error(f"Failed to load Anti-Spoofing model: {e}")

    def _crop_face(self, img_bgr, bbox, expansion=2.7):
        x, y, w, h = [int(v) for v in bbox]
        img_h, img_w = img_bgr.shape[:2]
        max_dim = max(w, h)
        cx, cy = x + w // 2, y + h // 2
        crop_size = int(max_dim * expansion)
        x1, y1 = cx - crop_size // 2, cy - crop_size // 2
        x2, y2 = x1 + crop_size, y1 + crop_size
        p_left, p_top = max(0, -x1), max(0, -y1)
        p_right, p_bottom = max(0, x2 - img_w), max(0, y2 - img_h)
        x1_src, y1_src = max(0, x1), max(0, y1)
        x2_src, y2_src = min(img_w, x2), min(img_h, y2)
        roi_src = img_bgr[y1_src:y2_src, x1_src:x2_src]
        if p_left > 0 or p_top > 0 or p_right > 0 or p_bottom > 0:
            roi = cv2.copyMakeBorder(roi_src, p_top, p_bottom, p_left, p_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            roi = roi_src
        return roi

    def _preprocess(self, crop_bgr):
        # ONNX model (anti_spoofing.onnx) được export từ pipe RGB [0,1] 
        # PyTorch model (MiniFASNet gốc) dùng BGR [0,1]
        if self.is_onnx:
            crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        else:
            crop = crop_bgr
            
        resized = cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0

        if not self.is_onnx:
            tensor_torch = torch.from_numpy(tensor).float()
            return tensor_torch.unsqueeze(0).to(self.device)
        
        return tensor

    def is_real(self, img_bgr, bbox):
        if self.session is None and self.torch_model is None:
            return True, 1.0

        # Cả ONNX (anti_spoofing.onnx) và PyTorch (MiniFASNetV2) đều dùng expansion 2.7
        expansion = 2.7
        crop = self._crop_face(img_bgr, bbox, expansion=expansion)
        if crop is None or crop.shape[0] < 5:
            return False, 0.0

        if self.is_onnx:
            tensor = self._preprocess(crop)
            batch = np.expand_dims(tensor, axis=0)
            try:
                logits = self.session.run(None, {self.input_name: batch})[0][0]
                real_logit = float(logits[0])
                spoof_logit = float(logits[1])
                logit_diff = real_logit - spoof_logit
                clipped = np.clip(logit_diff, -_LOGIT_CLIP, _LOGIT_CLIP)
                liveness_score = float(1.0 / (1.0 + np.exp(-clipped)))
                return liveness_score >= self.threshold, liveness_score
            except Exception as e:
                logger.error(f"ONNX Anti-spoofing error: {e}")
                return False, 0.0
        else:
            tensor = self._preprocess(crop)
            try:
                with torch.no_grad():
                    output = self.torch_model(tensor)
                    prob = F.softmax(output, dim=-1)
                    probs_list = prob[0].cpu().numpy().tolist()
                    
                    max_idx = int(np.argmax(probs_list))
                    
                    # Class mapping expected for MiniFASNet:
                    # 0: Fake (Printed paper)
                    # 1: Real
                    # 2: Fake (Video replay)
                    liveness_score = float(probs_list[1])
                    is_real = liveness_score >= self.threshold
                    
                    logger.info(f"ANTI_SPOOF: pred={max_idx}, real_prob={liveness_score:.4f}, pass={is_real} | Probs: [C0: {probs_list[0]:.4f}, C1: {probs_list[1]:.4f}, C2: {probs_list[2]:.4f}]")
                    return is_real, liveness_score
            except Exception as e:
                logger.error(f"PyTorch Anti-spoofing error: {e}")
                return False, 0.0
