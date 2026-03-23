"""
FaceRecognizer v5.1 - ArcFace ONNX wrapper.
Enhanced: Optimized ONNX provider selection (TensorRT > CUDA > OpenVINO > CPU),
          Prototype vector generation, Cosine-centroid outlier removal,
          A/B test dual-model support.
"""

import cv2
import numpy as np
import onnxruntime as ort
from config import (ARCFACE_PATH, ARCFACE_REF, TTA_ENABLED, 
                    OUTLIER_STD, OUTLIER_COSINE_MIN, OUTLIER_METHOD,
                    ONNX_PROVIDER_PRIORITY, ONNX_ENABLE_OPTIMIZATION,
                    ONNX_INTER_THREADS, ONNX_INTRA_THREADS)
from logger import get_logger

logger = get_logger("recognizer")


class FaceRecognizer:
    """ArcFace embedding extractor with optimized ONNX inference + TTA + Prototype."""

    def __init__(self, model_path=None):
        path = model_path or ARCFACE_PATH
        
        # --- ONNX Session Options (tối ưu hiệu suất) ---
        sess_opts = ort.SessionOptions()
        if ONNX_ENABLE_OPTIMIZATION:
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.inter_op_num_threads = ONNX_INTER_THREADS
        sess_opts.intra_op_num_threads = ONNX_INTRA_THREADS
        sess_opts.enable_mem_pattern = True
        sess_opts.enable_cpu_mem_arena = True
        
        # --- Provider Auto-Selection (chọn máy chạy nhanh nhất có sẵn) ---
        available = ort.get_available_providers()
        selected_providers = []
        
        for provider in ONNX_PROVIDER_PRIORITY:
            if provider in available:
                selected_providers.append(provider)
        
        # Luôn có CPU làm fallback cuối
        if "CPUExecutionProvider" not in selected_providers:
            selected_providers.append("CPUExecutionProvider")
        
        # Tạo session với provider tối ưu nhất
        self.session = ort.InferenceSession(path, sess_opts, providers=selected_providers)
        
        # Xác nhận provider đang dùng
        active_provider = self.session.get_providers()[0]
        provider_names = {
            "TensorrtExecutionProvider": "TensorRT GPU",
            "CUDAExecutionProvider": "CUDA GPU",
            "OpenVINOExecutionProvider": "OpenVINO",
            "DmlExecutionProvider": "DirectML GPU",
            "CPUExecutionProvider": "CPU",
        }
        self.device = provider_names.get(active_provider, active_provider)
        self.active_provider = active_provider
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        logger.info(f"ONNX providers available: {available}")
        logger.info(f"Selected: {active_provider} ({self.device})")
        if ONNX_ENABLE_OPTIMIZATION:
            logger.info(f"Graph optimization: ALL | Threads: inter={ONNX_INTER_THREADS} intra={ONNX_INTRA_THREADS}")

    def align(self, frame, lm5):
        """Align face to 112x112 using similarity transform."""
        tform, _ = cv2.estimateAffinePartial2D(lm5, ARCFACE_REF, method=cv2.LMEDS)
        if tform is None:
            return None
        return cv2.warpAffine(frame, tform, (112, 112), borderValue=0.0)

    def _preprocess(self, aligned_bgr):
        """Preprocess aligned face for ArcFace."""
        img = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = np.transpose((img - 127.5) / 127.5, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def _run(self, batch):
        """Run ArcFace inference."""
        return self.session.run([self.output_name], {self.input_name: batch})[0][0]

    def get_embedding(self, frame, lm5):
        """Extract normalized embedding with optional TTA."""
        aligned = self.align(frame, lm5)
        if aligned is None:
            return None

        if TTA_ENABLED:
            # Gom ảnh gốc và ảnh lật thành 1 batch (size=2) để chạy 1 lần duy nhất
            flipped = cv2.flip(aligned, 1)
            batch = np.concatenate([
                self._preprocess(aligned),
                self._preprocess(flipped)
            ], axis=0)
            
            # Chạy inference
            results = self.session.run([self.output_name], {self.input_name: batch})[0]
            
            # Cộng gộp 2 vector
            emb = results[0] + results[1]
        else:
            emb = self._run(self._preprocess(aligned))

        # L2 normalize
        emb = emb / np.linalg.norm(emb)
        return emb

    def get_embeddings_batch(self, frame, lm5_list):
        """Batch extract embeddings for multiple faces in ONE call.
        
        Khi có N mặt trong 1 frame → batch tất cả qua ONNX 1 lần thay vì N lần.
        Giảm latency đáng kể khi có nhiều mặt (~50ms → ~20ms cho 3 mặt).
        
        Args:
            frame: BGR image
            lm5_list: list of 5-point landmarks arrays
            
        Returns:
            list of normalized embeddings (or None for failed alignments)
        """
        aligned_list = []
        valid_indices = []
        
        for i, lm5 in enumerate(lm5_list):
            aligned = self.align(frame, lm5)
            if aligned is not None:
                aligned_list.append(aligned)
                valid_indices.append(i)
        
        if not aligned_list:
            return [None] * len(lm5_list)
        
        # Xây batch lớn: nếu TTA thì mỗi face = 2 ảnh (gốc + flip)
        batch_images = []
        for aligned in aligned_list:
            batch_images.append(self._preprocess(aligned))
            if TTA_ENABLED:
                batch_images.append(self._preprocess(cv2.flip(aligned, 1)))
        
        mega_batch = np.concatenate(batch_images, axis=0)
        
        # ONE ONNX call cho tất cả
        all_results = self.session.run([self.output_name], {self.input_name: mega_batch})[0]
        
        # Parse kết quả
        embeddings = [None] * len(lm5_list)
        stride = 2 if TTA_ENABLED else 1
        
        for idx, valid_i in enumerate(valid_indices):
            if TTA_ENABLED:
                emb = all_results[idx * 2] + all_results[idx * 2 + 1]
            else:
                emb = all_results[idx]
            emb = emb / np.linalg.norm(emb)
            embeddings[valid_i] = emb
        
        return embeddings

    # ==================== PROTOTYPE GENERATION (Điểm 4) ====================

    @staticmethod
    def compute_prototype(embeddings):
        """Tính vector Prototype đại diện cho 1 user.
        
        Prototype = trung bình cộng tất cả embeddings → L2 normalize.
        Đây là vector "sạch nhất" đại diện cho user, vì các nhiễu bị 
        triệt tiêu khi cộng trung bình nhiều góc khác nhau.
        
        Returns:
            prototype: numpy array 512-D normalized
        """
        if not embeddings:
            return None
        embs = np.array(embeddings, dtype=np.float32)
        proto = embs.mean(axis=0)
        proto = proto / np.linalg.norm(proto)
        return proto

    # ==================== OUTLIER REMOVAL (Điểm 6: Nâng cấp) ====================

    def clean_embeddings(self, embeddings, std_thresh=None):
        """Remove outlier embeddings using configurable method.
        
        Methods:
        - "std": Loại embeddings > N std from mean cosine sim (cũ)
        - "cosine_centroid": Loại embeddings có cosine < OUTLIER_COSINE_MIN so với centroid (mới)
        
        Args:
            embeddings: list of normalized embeddings
            std_thresh: override for std threshold
            
        Returns:
            cleaned list of embeddings
        """
        if len(embeddings) < 5:
            return embeddings

        embs = np.array(embeddings, dtype=np.float32)
        mean_emb = embs.mean(axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)

        # Cosine similarity to centroid
        sims = embs @ mean_emb

        if OUTLIER_METHOD == "cosine_centroid":
            # Phương pháp mới: chặn cứng bằng ngưỡng cosine tối thiểu
            mask = sims >= OUTLIER_COSINE_MIN
            cleaned = embs[mask].tolist()
            removed = len(embeddings) - len(cleaned)
            if removed > 0:
                logger.info(f"[Cosine-Centroid] Loai {removed} outlier (cosine < {OUTLIER_COSINE_MIN})")
                logger.info(f"Sim range: {sims.min():.3f} ~ {sims.max():.3f} (mean: {sims.mean():.3f})")
        else:
            # Phương pháp cũ: std-based
            thresh = std_thresh or OUTLIER_STD
            mean_sim = sims.mean()
            std_sim = sims.std()
            cutoff = mean_sim - thresh * std_sim
            mask = sims >= cutoff
            cleaned = embs[mask].tolist()
            removed = len(embeddings) - len(cleaned)
            if removed > 0:
                logger.info(f"[Std-Based] Loai {removed} outlier (< {cutoff:.3f} sim)")

        # Đảm bảo giữ lại ít nhất 3 embeddings
        return cleaned if len(cleaned) >= 3 else embeddings

    def select_best_embeddings(self, embeddings, scores, keep_top=15):
        """Select top-K embeddings by quality score + diversity.
        
        Strategy:
          1. Pair each embedding with its quality score
          2. Sort by quality score descending
          3. Keep top-K highest quality
          4. Run outlier removal on the kept set
          5. Generate prototype vector
        
        Returns:
            (selected_embeddings, selected_scores, prototype)
        """
        if len(embeddings) <= keep_top:
            cleaned = self.clean_embeddings(embeddings)
            prototype = self.compute_prototype(cleaned)
            return cleaned, scores[:len(cleaned)], prototype

        # Pair and sort by quality score descending
        paired = list(zip(embeddings, scores))
        paired.sort(key=lambda x: x[1], reverse=True)

        # Keep top-K
        top_embs = [p[0] for p in paired[:keep_top]]
        top_scores = [p[1] for p in paired[:keep_top]]

        # Clean outliers
        cleaned = self.clean_embeddings(top_embs)

        # Match scores to cleaned embeddings
        if len(cleaned) < len(top_embs):
            cleaned_set = set()
            for c in cleaned:
                for i, e in enumerate(top_embs):
                    if np.allclose(c, e, atol=1e-6) and i not in cleaned_set:
                        cleaned_set.add(i)
                        break
            final_scores = [top_scores[i] for i in sorted(cleaned_set)]
        else:
            final_scores = top_scores[:len(cleaned)]

        # Compute prototype
        prototype = self.compute_prototype(cleaned)

        logger.info(f"Quality filter: {len(embeddings)} → top {keep_top} → {len(cleaned)} final")
        logger.info(f"Score range: {min(final_scores):.3f} ~ {max(final_scores):.3f} (avg: {np.mean(final_scores):.3f})")
        if prototype is not None:
            proto_sims = np.array(cleaned) @ prototype
            logger.info(f"Prototype quality: sim range {proto_sims.min():.3f} ~ {proto_sims.max():.3f}")

        return cleaned, final_scores, prototype
