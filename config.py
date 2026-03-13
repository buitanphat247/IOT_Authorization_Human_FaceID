"""
Configuration constants for Face Recognition System.
"""
import os
import numpy as np

# === PATHS ===
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DB_DIR = os.path.join(ROOT_DIR, "db")
DATA_DIR = os.path.join(ROOT_DIR, "data", "enroll")
RECORDS_DIR = os.path.join(ROOT_DIR, "recordings")

ARCFACE_PATH = os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_l", "w600k_r50.onnx")
FL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
FL_PATH = os.path.join(MODELS_DIR, "face_landmarker.task")

# === CAMERA ===
CAMERA_ID = 0
CAMERA_W, CAMERA_H = 640, 480

# === FACE QUALITY ===
MIN_FACE_SIZE = 80
BLUR_THRESH = 50.0
MIN_BRIGHT, MAX_BRIGHT = 40, 220
MAX_TILT = 30
MIN_FACE_RATIO = 0.25   # min face_width / frame_width
MAX_FACE_RATIO = 0.50   # max face_width / frame_width

# === ENROLLMENT ===
ENROLL_INTERVAL = 0.35   # seconds between captures
ENROLL_KEEP_TOP = 15     # only keep top-N embeddings by quality (from ~28 captured)
ENROLL_STEPS = [
    ("NHIN THANG vao camera", 6, "straight"),
    ("QUAY DAU sang TRAI",    6, "left"),
    ("QUAY DAU sang PHAI",    6, "right"),
    ("NGANG NHE len TREN",    5, "up"),
    ("CUI NHE xuong DUOI",    5, "down"),
]

# === HEAD POSE THRESHOLDS ===
POSE_STRAIGHT_MAX = 0.12
POSE_LEFT_MIN = -0.22
POSE_RIGHT_MIN = 0.22
POSE_UP_MAX = 0.30
POSE_DOWN_MIN = 0.65

# === RECOGNITION ===
THRESHOLD = 0.45
TOP_K = 3                # top-K average for matching
TTA_ENABLED = True       # Test-Time Augmentation (flip horizontal)
OUTLIER_STD = 2.0        # remove embeddings > N std from mean during enrollment

# === ARCFACE ALIGNMENT ===
ARCFACE_REF = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]
], dtype=np.float32)

# Landmark indices: left_eye, right_eye, nose, left_mouth, right_mouth
LM_IDX_5PT = [33, 263, 1, 61, 291]

# === FACE MESH CONTOURS (for drawing) ===
FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]
LEFT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33]
RIGHT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]
LIPS = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61]
L_BROW = [70,63,105,66,107,55,65,52,53,46]
R_BROW = [300,293,334,296,336,285,295,282,283,276]
NOSE_BRIDGE = [168,6,197,195,5,4,1,19]
CONTOURS = [FACE_OVAL, LEFT_EYE, RIGHT_EYE, LIPS, L_BROW, R_BROW, NOSE_BRIDGE]
