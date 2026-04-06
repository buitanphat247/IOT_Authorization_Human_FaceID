import numpy as np
from scipy.optimize import linear_sum_assignment
from logger import get_logger

logger = get_logger("tracker")

def iou(bbox1, bbox2):
    """
    Tính Intersection over Union (IoU) giữa 2 bounding box.
    bbox định dạng: [x, y, w, h]
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate coordinates of intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    bb1_area = w1 * h1
    bb2_area = w2 * h2
    
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area + 1e-6)
    return iou


class Track:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox        # [x, y, w, h]
        self.hits = 1           # Số frame đã track được
        self.time_since_update = 0  # Số frame bị mất dấu
        self.is_activated = False
        
        # Có thể gán sau khi recognition chạy xong để giữ tên ổn định
        self.recognized_name = "Unknown"
        self.recognized_score = 0.0

    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        
    def mark_lost(self):
        self.time_since_update += 1


class ByteTracker:
    """
    ByteTrack implementation (Lightweight version).
    
    Thuật toán:
    - Nhận vào các detections có confidence score.
    - Chia detections làm 2 nhóm: High conf (D_high) và Low conf (D_low).
    - Lặp 1: Match các Tracks hiện tại với D_high bằng IoU.
    - Lặp 2: Match các Tracks đang bị mất dấu (lost) HOẶC chưa match ở Lặp 1 với D_low bằng IoU thấp hơn.
    - Tracks không match được sau Lặp 2 sẽ đánh dấu lost. 
    - D_high không match được sẽ sinh Track mới.
    - D_low không match được thì bỏ qua (noise).
    """
    def __init__(self, max_lost=30, iou_threshold=0.3, high_thresh=0.5, min_hits=3):
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold
        self.high_thresh = high_thresh
        self.min_hits = min_hits
        
        self.active_tracks = []
        self.next_id = 1

    def update(self, detections):
        """
        Cập nhật vị trí các track hiện tại.
        
        Args:
            detections: List of tuples (x, y, w, h, conf)
            
        Returns:
            List of Track objects đã được cập nhật.
        """
        if not detections:
            # Tăng time_since_update cho tất cả track, xóa track lost quá lâu
            for t in self.active_tracks:
                t.mark_lost()
            self.active_tracks = [t for t in self.active_tracks if t.time_since_update <= self.max_lost]
            return [t for t in self.active_tracks if t.is_activated]

        # Phân loại high/low score detections
        dets_high = []
        dets_low = []
        for i, det in enumerate(detections):
            x, y, w, h, conf = det
            if conf >= self.high_thresh:
                dets_high.append((i, det))
            else:
                dets_low.append((i, det))

        unmatched_tracks = list(range(len(self.active_tracks)))
        
        # --- LẶP 1: MATCH D_HIGH vs TẤT CẢ TRACKS ---
        unmatched_dets_high = []
        if len(self.active_tracks) > 0 and len(dets_high) > 0:
            cost_matrix = np.zeros((len(self.active_tracks), len(dets_high)))
            for t_idx, track in enumerate(self.active_tracks):
                for d_idx, (_, det) in enumerate(dets_high):
                    # Cost = 1 - IoU 
                    cost_matrix[t_idx, d_idx] = 1 - iou(track.bbox, det[:4])
                    
            # Dùng Hungarian algorithm (linear_sum_assignment) để tìm cặp ghép tốt nhất
            row_inds, col_inds = linear_sum_assignment(cost_matrix)
            
            # Lọc các cặp không thỏa IoU
            for r, c in zip(row_inds, col_inds):
                if cost_matrix[r, c] <= (1 - self.iou_threshold):
                    # Hợp lệ
                    d_original_idx, det_data = dets_high[c]
                    self.active_tracks[r].update(det_data[:4])
                    if r in unmatched_tracks:
                        unmatched_tracks.remove(r)
                    dets_high[c] = (d_original_idx, None) # Đánh dấu đã dùng
                
        # Gom các dets_high chưa dùng lại
        unmatched_dets_high = [det for det in dets_high if det[1] is not None]

        # --- LẶP 2: MATCH D_LOW vs TRACKS CÒN LẠI THIẾU MATCH ---
        unmatched_tracks_second_pass = []
        if len(unmatched_tracks) > 0 and len(dets_low) > 0:
            remaining_tracks = [self.active_tracks[t] for t in unmatched_tracks]
            cost_matrix_low = np.zeros((len(remaining_tracks), len(dets_low)))
            for t_idx, track in enumerate(remaining_tracks):
                for d_idx, (_, det) in enumerate(dets_low):
                    # Thường Lặp 2 sẽ hạ IoU threshold (e.g. 0.5 * threshold) để dễ nhận lại mặt bị mờ
                    cost_matrix_low[t_idx, d_idx] = 1 - iou(track.bbox, det[:4])
            
            row_inds, col_inds = linear_sum_assignment(cost_matrix_low)
            
            for r, c in zip(row_inds, col_inds):
                # IoU threshold χα lỏng hơn cho Lặp 2 chút ít (vd: + 0.1)
                if cost_matrix_low[r, c] <= (1 - max(0.1, self.iou_threshold - 0.1)):
                    t_original_idx = unmatched_tracks[r]
                    d_original_idx, det_data = dets_low[c]
                    self.active_tracks[t_original_idx].update(det_data[:4])
                    unmatched_tracks_second_pass.append(t_original_idx)
                    dets_low[c] = (d_original_idx, None)
                    
        # Loại các track đã match ở Lặp 2
        final_unmatched_tracks = [t for t in unmatched_tracks if t not in unmatched_tracks_second_pass]

        # --- XỬ LÝ KẾT QUẢ ---
        # 1. Các track không match được -> mark lost
        for t_idx in final_unmatched_tracks:
            self.active_tracks[t_idx].mark_lost()
            
        # 2. Tracks lost quá lâu -> Xóa
        self.active_tracks = [t for t in self.active_tracks if t.time_since_update <= self.max_lost]
        
        # 3. Các high conf detections dư -> Khởi tạo Track Mới
        for _, det in unmatched_dets_high:
            if det is not None:
                new_track = Track(self.next_id, det[:4])
                self.active_tracks.append(new_track)
                self.next_id += 1
                if self.next_id > 99999: # Chống tràn id
                    self.next_id = 1
                    
        # Active các track đạt min_hits
        for track in self.active_tracks:
            if not track.is_activated and track.hits >= self.min_hits:
                track.is_activated = True

        return [t for t in self.active_tracks if t.is_activated or t.time_since_update == 0]
