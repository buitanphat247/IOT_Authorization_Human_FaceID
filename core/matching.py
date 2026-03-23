"""
MatchingEngine v5.3 - Open-Set Face Recognition.
Anti-False-Accept: absolute threshold + top1-top2 margin + quality gate.

Strategies:
  1. FAISS K-NN search: top-K group averaging per user
  2. Prototype matching: O(1) matrix multiply (optional, disabled by default)
  3. Open-set rejection: unknown_threshold + margin check
"""

import numpy as np
from collections import defaultdict
from logger import get_logger

logger = get_logger("matching")


class MatchingEngine:
    """Open-set face matching engine with anti-false-accept.
    
    Key difference from closed-set: we can return "Unknown" 
    even if there IS a best match — if the match isn't confident enough.
    
    Usage:
        engine = MatchingEngine(top_k=3, unknown_threshold=0.40)
        name, score = engine.match(
            query_emb, faiss_index, id_to_name_fn,
            proto_matrix, proto_names
        )
    """

    def __init__(self, top_k=3, proto_weight=0.4, proto_enabled=False,
                 unknown_threshold=0.40, margin_threshold=0.05):
        self.top_k = top_k
        self.proto_weight = proto_weight
        self.proto_enabled = proto_enabled
        # --- Open-set parameters ---
        self.unknown_threshold = unknown_threshold  # Absolute: score < this → Unknown
        self.margin_threshold = margin_threshold    # top1 - top2 < this → Unknown (ambiguous)

    def match(self, query_emb, faiss_index, id_to_name_fn,
              proto_matrix=None, proto_names=None):
        """Find best matching user via open-set matching.
        
        Returns:
            (name: str, score: float)
            name = "Unknown" if no confident match
        """
        if faiss_index.ntotal == 0:
            return "Unknown", 0.0

        # --- Strategy 1: Top-K raw FAISS matching (primary) ---
        k = min(self.top_k * 5, faiss_index.ntotal)
        scores, ids = faiss_index.search(
            np.array([query_emb], dtype=np.float32), k
        )

        # Resolve FAISS IDs to user names
        valid_ids = [int(eid) for eid in ids[0] if eid != -1]
        id_name_map = id_to_name_fn(valid_ids) if valid_ids else {}

        user_scores = defaultdict(list)
        for score, eid in zip(scores[0], ids[0]):
            if eid == -1:
                continue
            name = id_name_map.get(int(eid))
            if name:
                user_scores[name].append(float(score))

        # Top-K average per user
        user_avg_scores = {}
        for name, sc_list in user_scores.items():
            avg = float(np.mean(sorted(sc_list, reverse=True)[:self.top_k]))
            user_avg_scores[name] = avg

        if not user_avg_scores:
            return "Unknown", 0.0

        # Sort by score descending
        sorted_users = sorted(user_avg_scores.items(), key=lambda x: x[1], reverse=True)
        topk_name, topk_score = sorted_users[0]
        
        # Get second-best score for margin check
        second_best_score = sorted_users[1][1] if len(sorted_users) > 1 else 0.0

        # --- Strategy 2: Prototype matching (optional, disabled by default) ---
        proto_name, proto_score = "Unknown", 0.0
        if self.proto_enabled and proto_matrix is not None and len(proto_names) > 0:
            query = np.array(query_emb, dtype=np.float32)
            sims = proto_matrix @ query
            best_idx = int(np.argmax(sims))
            proto_name = proto_names[best_idx]
            proto_score = float(sims[best_idx])

        # --- Fusion ---
        if self.proto_enabled and proto_name != "Unknown":
            w = self.proto_weight
            if proto_name == topk_name:
                final_score = w * proto_score + (1 - w) * topk_score
                final_name = proto_name
            else:
                if proto_score >= topk_score:
                    final_score = proto_score
                    final_name = proto_name
                else:
                    final_score = topk_score
                    final_name = topk_name
        else:
            final_score = topk_score
            final_name = topk_name

        # ========================================
        # === OPEN-SET REJECTION GATES ===
        # ========================================

        # Gate 1: Absolute threshold — score quá thấp → Unknown
        if final_score < self.unknown_threshold:
            logger.debug(
                f"REJECT (absolute): {final_name} score={final_score:.4f} "
                f"< threshold={self.unknown_threshold}"
            )
            return "Unknown", final_score

        # Gate 2: Top1-Top2 margin — nếu score top1 và top2 quá gần nhau
        # → ambiguous, không chắc user nào → Unknown
        if len(sorted_users) > 1:
            margin = topk_score - second_best_score
            if margin < self.margin_threshold and final_score < 0.70:
                logger.info(
                    f"REJECT (margin): {final_name} score={final_score:.4f}, "
                    f"margin={margin:.4f} < {self.margin_threshold} (ambiguous)"
                )
                return "Unknown", final_score

        logger.debug(f"MATCH: {final_name} score={final_score:.4f}")
        return final_name, final_score
