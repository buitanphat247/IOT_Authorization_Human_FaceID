"""
Configuration constants for Face Recognition System v5.1
Enhanced: 2-Stage Threshold, Quality-Weighted Voting, Prototype Matching,
          Landmark Stability Gating, A/B Test Framework,
          ONNX Optimized Inference, Multi-Threaded Pipeline, Score Stability.
"""
import os
import numpy as np

# === PATHS ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DB_DIR = os.path.join(ROOT_DIR, "db")
DATA_DIR = os.path.join(ROOT_DIR, "data", "enroll")
RECORDS_DIR = os.path.join(ROOT_DIR, "recordings")

# Model pretrained (dự phòng)
ARCFACE_PATH_PRETRAINED = os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_l", "w600k_r50.onnx")

# === HOT-SWAP MODEL (Tự động chọn model tốt nhất có sẵn) ===
# Ưu tiên: V5 fine-tuned > V4 > Pretrained w600k_r50
# Chỉ cần drop file .onnx vào thư mục models/ là hệ thống tự nhận!
_MODEL_PRIORITY = [
    os.path.join(MODELS_DIR, "arcface_best_model_v5.onnx"),   # V5 (đang train)
    os.path.join(MODELS_DIR, "arcface_best_model_v4.onnx"),   # V4 (mode collapse, backup)
    ARCFACE_PATH_PRETRAINED,                                   # Pretrained (mặc định)
]

ARCFACE_PATH = None
for _candidate in _MODEL_PRIORITY:
    if os.path.exists(_candidate):
        ARCFACE_PATH = _candidate
        break

if ARCFACE_PATH is None:
    ARCFACE_PATH = ARCFACE_PATH_PRETRAINED  # Fallback cuối cùng

# Log model đang dùng
import sys
_model_name = os.path.basename(ARCFACE_PATH)
if "v5" in _model_name:
    print(f"[CONFIG] 🌟 ArcFace Model: {_model_name} (V5 Fine-tuned)")
elif "v4" in _model_name:
    print(f"[CONFIG] ⚠️ ArcFace Model: {_model_name} (V4 - Mode Collapse risk)")
else:
    print(f"[CONFIG] ✅ ArcFace Model: {_model_name} (Pretrained w600k_r50)")

# === FACE DETECTOR BACKEND ===
# "scrfd"    : SCRFD (chính xác, tốt cho recognition) — CẦN insightface
# "mediapipe": MediaPipe FaceLandmarker (nhẹ, nhanh) — hiện tại
# "hybrid"   : SCRFD primary + MediaPipe fallback khi SCRFD fail
DETECTOR_BACKEND = "hybrid"

# SCRFD Config
SCRFD_MODEL = "buffalo_l"           # buffalo_l (tốt nhất) hoặc buffalo_sc (nhẹ nhất)
SCRFD_CTX_ID = -1                   # -1 = CPU, 0 = GPU
SCRFD_DET_SIZE = (640, 640)         # Detection input size
SCRFD_DET_THRESH = 0.5              # Detection confidence threshold

# === BYTETRACK TRACKER (Thay Centroid Tracker) ===
# Giữ ID ổn định khi nhiều người qua lại, re-ID khi mặt tạm mất
TRACKER_ENABLED = True              # True = ByteTrack, False = Centroid cũ
TRACKER_MAX_LOST = 30               # Số frame giữ track khi mất mặt (~1s @ 30fps)
TRACKER_IOU_THRESHOLD = 0.3         # IoU tối thiểu để match track-detection
TRACKER_HIGH_THRESH = 0.5           # Confidence cao cho Stage 1
TRACKER_MIN_HITS = 3                # Số frame tối thiểu để confirm track

FL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
FL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

# === ANTI-SPOOFING (MiniFASNet) ===
# Chuyển về sử dụng ONNX để khắc phục lỗi inference kiến trúc PyTorch làm model luôn trả class 2.
ANTI_SPOOF_PATH = os.path.join(MODELS_DIR, "anti_spoofing.onnx")
ANTI_SPOOF_THRESHOLD = 0.5   # Nâng ngưỡng nếu cần (0.5 là chuẩn, <0.5 là Fake)

# === CAMERA ===
CAMERA_ID = 0
CAMERA_W, CAMERA_H = 640, 480

# === FACE QUALITY (Gating Đầu Vào) ===
MIN_FACE_SIZE = 50         # Giảm từ 80 — webcam browser có những frame nhỏ
BLUR_THRESH = 30.0         # Giảm từ 50 — webcam browser thường bị blur hơn
MIN_BRIGHT, MAX_BRIGHT = 30, 230
MAX_TILT = 35              # Tăng từ 30 — cho phép nghiêng hơn
MIN_FACE_RATIO = 0.12      # Giảm từ 0.25 — cho phép mặt nhỏ hơn
MAX_FACE_RATIO = 0.65      # Tăng từ 0.50

# === ENHANCED GATING (Điểm 7: Landmark Stability + Eye Openness) ===
GATING_EYE_OPENNESS_MIN = 0.015      # Tỷ lệ mở mắt tối thiểu (chặn nhắm mắt / khuất mắt)
GATING_BBOX_JERK_MAX = 80.0          # Pixel di chuyển tối đa BBox giữa 2 frame (chặn rung lắc)
GATING_LANDMARK_STABILITY = True     # Bật kiểm tra ổn định landmark liên frame

# === ENROLLMENT ===
ENROLL_INTERVAL = 0.35   # seconds between captures
ENROLL_KEEP_TOP = 30     # Lưu toàn bộ 30 vector đặc điểm tốt nhất vào Database thay vì 15
ENROLL_STEPS = [
    ("Quet khuon mat (Giu yen hoac nghieng nhe)", 30, "any"),
]

# === HEAD POSE THRESHOLDS ===
POSE_STRAIGHT_MAX = 0.12
POSE_LEFT_MIN = -0.22
POSE_RIGHT_MIN = 0.22
POSE_UP_MAX = 0.30
POSE_DOWN_MIN = 0.65

# === RECOGNITION — ƯU TIÊN 1: Decision Logic ===
# --- DYNAMIC THRESHOLD (Ngưỡng động theo Benchmark Vàng của Pretrained) ---
# Theo đồ thị, Imposter max là ~0.20, Genuine min là ~0.30. Optimal = 0.23.
# Để an toàn cho Camera (Chống False Accept tuyệt đối), ta nhích nhẹ lên mức 0.30 - 0.35
THRESHOLD_ACCEPT_HIGH_QUALITY = 0.38  # Ngưỡng an toàn bảo mật cao (Bỏ 0.67 cũ)
THRESHOLD_ACCEPT_LOW_QUALITY = 0.34   # Chấp nhận mờ/thấp nhưng yêu cầu nháy mắt 
THRESHOLD_REJECT = 0.25               # Dưới 0.25 Cảnh báo Đỏ (Unknown)
# Giữa REJECT < score < ACCEPT → trạng thái UNCERTAIN, cần thêm frame
THRESHOLD = 0.38          # Giữ lại không đổi API cũ
THRESHOLD_ACCEPT = THRESHOLD_ACCEPT_HIGH_QUALITY # Backward compatibility

TOP_K = 3                 # faiss search top-K group by user
TTA_ENABLED = True        # Test-Time Augmentation (flip horizontal)

# === OUTLIER REMOVAL (Điểm 6: Cosine-to-Centroid thay vì chỉ std) ===
OUTLIER_STD = 2.0                    # Ngưỡng std cũ (fallback)
OUTLIER_COSINE_MIN = 0.45            # Hạ xuống 0.45 để phù hợp với đồ thị Cosine của Model Pretrained (Genuine min ~ 0.3)
OUTLIER_METHOD = "cosine_centroid"   # "std" hoặc "cosine_centroid"

# === MULTI-FRAME VOTING (Điểm 3: Quality-Weighted thay vì Avg đều tay) ===
MULTI_FRAME_ENABLED = True
MULTI_FRAME_BUFFER = 7              # Tăng buffer lên 7 frame
MULTI_FRAME_TOP_N = 5               # Chỉ lấy top-5 frame tốt nhất trong buffer
MULTI_FRAME_WEIGHTED = True          # Bật quality-weighted voting

# === PROTOTYPE MATCHING (Cải tiến: Cascaded Rejection — chỉ có quyền CHẶN) ===
# Prototype = vector trung bình của mỗi user. 
# FAISS nhanh (nhưng có thể nhầm due to noise point) → Prototype verify lại.
PROTOTYPE_ENABLED = True
PROTOTYPE_WEIGHT = 0.4
PROTOTYPE_MODE = "reject_only"       # MỚI v5.4: Chỉ được quyền CHẶN
PROTOTYPE_REJECT_THRESHOLD = 0.20    # Prototype cosine < 0.20 → CHẶN (ngưỡng thấp, an toàn)

# === COHORT NORMALIZATION (Ngừa Model Sụp Đổ) ===
# Khắc phục hiện tượng 1 khuôn mặt "quá chung chung" match bậy với tất cả mọi người.
# Tính Z-Score so với top-N kẻ mạo danh gần nhất.
COHORT_ENABLED = True
COHORT_Z_THRESHOLD = 2.0             # Z-score < 2.0 (thuộc về đám đông) → CHẶN

# === 3-STATE OUTPUT (Điểm 9: accepted / unknown / low_quality) ===
THREE_STATE_ENABLED = True

# === A/B TEST FRAMEWORK (Điểm 8: So sánh model cũ vs mới) ===
AB_TEST_ENABLED = False              # TẮT trên production (không có model pretrained trên server)
AB_TEST_MODEL_B_PATH = os.path.join(MODELS_DIR, "arcface_best_model_v3.onnx")  # Fallback: model cũ

# === DATABASE BACKEND (BUG-03: Chọn FAISS local vs pgvector cloud) ===
# "faiss": FAISS local (mặc định, tốt cho < 10,000 faces)
# "pgvector": Supabase pgvector (cloud-native, scalable, cần enable extension)
DB_BACKEND = "faiss"

# === ONNX FP16 MODEL (Task 11: Tối ưu inference) ===
ARCFACE_FP16_PATH = os.path.join(MODELS_DIR, "arcface_best_model_v4_fp16.onnx")

# === ONNX RUNTIME OPTIMIZATION (Tối ưu Inference Engine) ===
# Thứ tự ưu tiên provider: TensorRT > CUDA > OpenVINO > DirectML > CPU
ONNX_PROVIDER_PRIORITY = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "OpenVINOExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]
ONNX_ENABLE_OPTIMIZATION = True      # Bật tối ưu graph ONNX
ONNX_INTER_THREADS = 2               # Số thread tính song song giữa các node
ONNX_INTRA_THREADS = 4               # Số thread tính toán bên trong mỗi node

# === MULTI-THREADED PIPELINE (Tách luồng Camera / Detect / Recognize) ===
PIPELINE_THREADED = True             # Bật pipeline đa luồng
PIPELINE_QUEUE_SIZE = 2              # Kích thước hàng đợi giữa các tầng
FRAME_SKIP = 2                       # Skip N frame giữa các lần recognize (0=không skip)

# === SCORE STABILITY / TEMPORAL SMOOTHING (Decision State Machine) ===
# Chống nhấp nháy kết quả bằng EMA (Exponential Moving Average)
SCORE_SMOOTHING_ENABLED = True
SCORE_SMOOTHING_ALPHA = 0.6          # Hệ số EMA (tăng từ 0.4 → phản ứng nhanh hơn khi đổi mặt)
SCORE_STABLE_FRAMES = 4             # Phải ổn định N frame liên tiếp mới chốt ACCEPT (tăng từ 3)
SCORE_STABLE_TOLERANCE = 0.06       # Score dao động < 0.06 = ổn định (siết chặt từ 0.08)

# === ARCFACE ALIGNMENT ===
ARCFACE_REF = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]
], dtype=np.float32)

# Landmark indices: left_eye, right_eye, nose, left_mouth, right_mouth
LM_IDX_5PT = [33, 263, 1, 61, 291]

# Eye landmark indices for openness check (Điểm 7)
LEFT_EYE_TOP = 159       # Mí trên mắt trái
LEFT_EYE_BOTTOM = 145    # Mí dưới mắt trái
RIGHT_EYE_TOP = 386      # Mí trên mắt phải
RIGHT_EYE_BOTTOM = 374   # Mí dưới mắt phải

# === BLINK THRESHOLDS cho Dynamic Threshold ===
BLINK_EAR_CLOSED = 0.025   # Dưới mức này xem là mắt nhắm nháy mắt
BLINK_EAR_OPEN = 0.045     # Trên mức này xem là mắt đang mở bình thường

# === FACE MESH CONTOURS (for drawing) ===
FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]
LEFT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33]
RIGHT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]
LIPS = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61]
L_BROW = [70,63,105,66,107,55,65,52,53,46]
R_BROW = [300,293,334,296,336,285,295,282,283,276]
NOSE_BRIDGE = [168,6,197,195,5,4,1,19]
CONTOURS = [FACE_OVAL, LEFT_EYE, RIGHT_EYE, LIPS, L_BROW, R_BROW, NOSE_BRIDGE]
