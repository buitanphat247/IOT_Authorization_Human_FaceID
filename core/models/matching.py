"""
MatchingEngine v5.11 - Open-Set Face Recognition.
Anti-False-Accept: absolute threshold + top1-top2 margin + quality gate.
Cascaded Rejection: Prototype as rejection-only filter (v5.4 upgrade).
Cohort Normalization: Z-score calibration chống "ai cũng ra mặt tôi" (v5.11).

Strategies:
  1. FAISS K-NN search: top-K group averaging per user
  2. Prototype rejection: cosine check → can only REJECT, never boost
  3. Open-set rejection: unknown_threshold + margin check
  4. Cohort Normalization: score phải nổi bật giữa n kẻ mạo danh gần nhất (Z-score)
"""

import numpy as np
from collections import defaultdict
from logger import get_logger

logger = get_logger("matching")


class MatchingEngine:
    """Open-set face matching engine with anti-false-accept.
    
    v5.4: Prototype Cascaded Rejection
    - Prototype chỉ có quyền CHẶN, không có quyền nâng score
    - Nếu FAISS match OK nhưng Prototype score quá thấp → REJECT
    
    v5.11: Cohort Normalization (Z-Score)
    - Tính Z-score từ top-N kẻ mạo danh gần nhất
    - Giúp triệt tiêu lỗi false match do model bị sụp (mode collapse) ở 1 số khuôn mặt
    """

    def __init__(self, top_k=3, proto_weight=0.4, proto_enabled=False,
                 proto_mode="reject_only", proto_reject_threshold=0.20,
                 unknown_threshold=0.50, margin_threshold=0.05,
                 cohort_enabled=True, cohort_z_threshold=2.0):
        self.top_k = top_k
        self.proto_weight = proto_weight
        self.proto_enabled = proto_enabled
        self.proto_mode = proto_mode                    # "reject_only" hoặc "fusion"
        self.proto_reject_threshold = proto_reject_threshold  # Prototype < này → CHẶN
        # --- Open-set parameters ---
        self.unknown_threshold = unknown_threshold
        self.margin_threshold = margin_threshold
        # --- Cohort Normalization ---
        self.cohort_enabled = cohort_enabled
        self.cohort_z_threshold = cohort_z_threshold

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
        # Lấy nhiều kết quả hơn (min 30) để tạo đủ tập "kẻ mạo danh" (Cohort)
        k_search = max(self.top_k * 5, 30) if self.cohort_enabled else self.top_k * 5
        k = min(k_search, faiss_index.ntotal)
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

        # --- Strategy 2: Prototype Handling ---
        proto_name, proto_score = "Unknown", 0.0
        if self.proto_enabled and proto_matrix is not None and len(proto_names) > 0:
            query = np.array(query_emb, dtype=np.float32)
            sims = proto_matrix @ query
            best_idx = int(np.argmax(sims))
            proto_name = proto_names[best_idx]
            proto_score = float(sims[best_idx])

        # --- Final score: FAISS only (Prototype không inflate score nữa!) ---
        final_score = topk_score
        final_name = topk_name

        # Legacy fusion mode (chỉ dùng khi cố tình bật "fusion")
        if self.proto_enabled and self.proto_mode == "fusion" and proto_name != "Unknown":
            w = self.proto_weight
            if proto_name == topk_name:
                final_score = w * proto_score + (1 - w) * topk_score
                final_name = proto_name
            else:
                if proto_score >= topk_score:
                    final_score = proto_score
                    final_name = proto_name

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

        # Gate 3: Single-user penalty — khi DB chỉ có 1 user,
        # không có top2 để so margin → dễ False Accept.
        if len(sorted_users) == 1 and final_score < 0.60:
            logger.info(
                f"REJECT (single-user): {final_name} score={final_score:.4f} "
                f"< 0.60 (no second user to compare margin)"
            )
            return "Unknown", final_score

        # Gate 4: Cohort Normalization (Z-Score)
        # So sánh khoảng cách của top1 so với "đám đông" mạo danh.
        if self.cohort_enabled and len(sorted_users) >= 3:
            # Lấy tới 10 kẻ mạo danh gần nhất (cohort)
            cohort_scores = [sc for _, sc in sorted_users[1:11]]
            if len(cohort_scores) >= 2:
                mu = np.mean(cohort_scores)
                std = np.std(cohort_scores) + 1e-6  # Tránh chia 0
                z_score = (final_score - mu) / std
                
                # Z-score < threshold mang nghĩa top1 KHÔNG cách biệt khỏi đám đông
                # CHỈ áp dụng chặt đứt kết quả cho nhóm score vùng tranh tối tranh sáng (<0.7)
                if z_score < self.cohort_z_threshold and final_score < 0.70:
                    logger.info(
                        f"REJECT (cohort): {final_name} score={final_score:.4f}, "
                        f"z={z_score:.2f} < {self.cohort_z_threshold} (mu={mu:.4f})"
                    )
                    return "Unknown", final_score

        # ========================================
        # === Gate 5: PROTOTYPE CASCADED REJECTION (v5.4) ===
        # ========================================
        # Prototype chỉ có quyền CHẶN, không bao giờ có quyền mở cửa.
        # Nếu FAISS đã match OK, nhưng Prototype nói "Mặt này không giống" → CHẶN!
        if (self.proto_enabled and self.proto_mode == "reject_only" 
                and proto_matrix is not None and len(proto_names) > 0):
            # Tìm prototype score CHO ĐÚNG NGƯỜI mà FAISS đã match
            if final_name in proto_names:
                name_idx = proto_names.index(final_name)
                matched_proto_score = float(
                    proto_matrix[name_idx] @ np.array(query_emb, dtype=np.float32)
                )
                
                if matched_proto_score < self.proto_reject_threshold:
                    logger.info(
                        f"REJECT (prototype): FAISS said {final_name} "
                        f"(score={final_score:.4f}), but Prototype disagrees "
                        f"(proto_score={matched_proto_score:.4f} "
                        f"< {self.proto_reject_threshold})"
                    )
                    return "Unknown", final_score

        logger.debug(f"MATCH: {final_name} score={final_score:.4f}")
        return final_name, final_score
