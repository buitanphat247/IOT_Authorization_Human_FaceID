"""
FaceRecognizer - ArcFace ONNX wrapper.
Handles alignment, embedding extraction, TTA (Test-Time Augmentation).
"""

import cv2
import numpy as np
import onnxruntime as ort
from config import ARCFACE_PATH, ARCFACE_REF, TTA_ENABLED


class FaceRecognizer:
    """ArcFace embedding extractor with TTA for higher accuracy."""

    def __init__(self):
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            self.device = "GPU"
            self.session = ort.InferenceSession(
                ARCFACE_PATH,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
        else:
            self.device = "CPU"
            self.session = ort.InferenceSession(
                ARCFACE_PATH,
                providers=['CPUExecutionProvider']
            )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

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
        """Extract normalized embedding with optional TTA.

        TTA: run inference on original + horizontally flipped face,
        then average both embeddings. This improves robustness to
        slight pose and lighting asymmetry.
        """
        aligned = self.align(frame, lm5)
        if aligned is None:
            return None

        emb = self._run(self._preprocess(aligned))

        if TTA_ENABLED:
            # Flip horizontally and re-extract
            flipped = cv2.flip(aligned, 1)
            emb_flip = self._run(self._preprocess(flipped))
            # Average both embeddings
            emb = emb + emb_flip

        # L2 normalize
        emb = emb / np.linalg.norm(emb)
        return emb

    def clean_embeddings(self, embeddings, std_thresh=2.0):
        """Remove outlier embeddings that are too far from the mean.
        This improves enrollment quality by removing bad captures.
        
        Args:
            embeddings: list of normalized embeddings
            std_thresh: remove embeddings > N std from mean cosine sim
            
        Returns:
            cleaned list of embeddings
        """
        if len(embeddings) < 5:
            return embeddings

        embs = np.array(embeddings)
        mean_emb = embs.mean(axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)

        # Cosine similarity to mean
        sims = embs @ mean_emb
        mean_sim = sims.mean()
        std_sim = sims.std()

        # Keep only within threshold
        mask = sims >= (mean_sim - std_thresh * std_sim)
        cleaned = embs[mask].tolist()

        removed = len(embeddings) - len(cleaned)
        if removed > 0:
            print(f"    -> Loai {removed} outlier (< {mean_sim - std_thresh * std_sim:.3f} sim)")

        return cleaned if len(cleaned) >= 3 else embeddings

    def select_best_embeddings(self, embeddings, scores, keep_top=15):
        """Select top-K embeddings by quality score + diversity.
        
        Strategy:
          1. Pair each embedding with its quality score
          2. Sort by quality score descending
          3. Keep top-K highest quality
          4. Then run outlier removal on the kept set
          5. L2 normalize final set
        
        Args:
            embeddings: list of embedding vectors
            scores: list of quality scores (same length)
            keep_top: max embeddings to keep
            
        Returns:
            (selected_embeddings, selected_scores)
        """
        if len(embeddings) <= keep_top:
            # Already fewer than limit, just clean outliers
            cleaned = self.clean_embeddings(embeddings)
            return cleaned, scores[:len(cleaned)]

        # Pair and sort by quality score descending
        paired = list(zip(embeddings, scores))
        paired.sort(key=lambda x: x[1], reverse=True)

        # Keep top-K
        top_embs = [p[0] for p in paired[:keep_top]]
        top_scores = [p[1] for p in paired[:keep_top]]

        # Clean outliers from the top set
        cleaned = self.clean_embeddings(top_embs)

        # Match scores to cleaned embeddings
        # (clean_embeddings may remove some, keep corresponding scores)
        if len(cleaned) < len(top_embs):
            # Re-match by finding which embeddings survived
            cleaned_set = set()
            for c in cleaned:
                for i, e in enumerate(top_embs):
                    if np.allclose(c, e, atol=1e-6) and i not in cleaned_set:
                        cleaned_set.add(i)
                        break
            final_scores = [top_scores[i] for i in sorted(cleaned_set)]
        else:
            final_scores = top_scores[:len(cleaned)]

        print(f"    -> Quality filter: {len(embeddings)} → top {keep_top} → {len(cleaned)} final")
        print(f"    -> Score range: {min(final_scores):.3f} ~ {max(final_scores):.3f} (avg: {np.mean(final_scores):.3f})")

        return cleaned, final_scores
