"""
HybridDetector v6.0 — SCRFD Primary + MediaPipe Fallback.
Factory function to create the right detector based on config.

Pipeline:
  Frame → SCRFD detect
    ├─ OK → return faces
    └─ FAIL (no faces / insightface not installed)
         → MediaPipe fallback → return faces

Usage:
    from models.hybrid_detector import create_detector
    detector = create_detector()
    faces = detector.detect(frame_rgb)
"""

from logger import get_logger

logger = get_logger("hybrid_detector")


def create_detector(mode="video", num_faces=3):
    """Factory: tạo detector dựa trên config DETECTOR_BACKEND.
    
    Returns:
        detector object with .detect(frame_rgb) method
    """
    from config import DETECTOR_BACKEND
    
    if DETECTOR_BACKEND == "scrfd":
        return _create_scrfd(num_faces)
    elif DETECTOR_BACKEND == "mediapipe":
        return _create_mediapipe(mode, num_faces)
    elif DETECTOR_BACKEND == "hybrid":
        return HybridDetector(mode=mode, num_faces=num_faces)
    else:
        logger.warning(f"Unknown DETECTOR_BACKEND='{DETECTOR_BACKEND}', fallback to mediapipe")
        return _create_mediapipe(mode, num_faces)


def _create_scrfd(num_faces):
    """Try to create SCRFD detector."""
    try:
        from models.scrfd_detector import SCRFDDetector, INSIGHTFACE_AVAILABLE
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("insightface not available")
        detector = SCRFDDetector(num_faces=num_faces)
        logger.info("✅ Detector: SCRFD (InsightFace)")
        print("[CONFIG] 🎯 Face Detector: SCRFD (InsightFace) — Production Grade")
        return detector
    except Exception as e:
        logger.warning(f"SCRFD init failed: {e}. Falling back to MediaPipe.")
        print(f"[CONFIG] ⚠️ SCRFD not available ({e}), using MediaPipe")
        from models.detector import FaceDetector
        return FaceDetector(mode="image", num_faces=num_faces)


def _create_mediapipe(mode, num_faces):
    """Create MediaPipe detector."""
    from models.detector import FaceDetector
    logger.info("✅ Detector: MediaPipe FaceLandmarker")
    print("[CONFIG] ✅ Face Detector: MediaPipe FaceLandmarker")
    return FaceDetector(mode=mode, num_faces=num_faces)


class HybridDetector:
    """SCRFD Primary + MediaPipe Fallback.
    
    - Dùng SCRFD cho mọi frame (chính xác hơn, landmark chuẩn hơn)
    - Nếu SCRFD không detect được (confidence thấp, mặt nghiêng quá) → MediaPipe cứu
    - Nếu insightface chưa cài → tự động dùng MediaPipe 100%
    """

    def __init__(self, mode="video", num_faces=3):
        self._scrfd = None
        self._mediapipe = None
        self._scrfd_available = False
        self._num_faces = num_faces
        
        # Try SCRFD first
        try:
            from models.scrfd_detector import SCRFDDetector, INSIGHTFACE_AVAILABLE
            if INSIGHTFACE_AVAILABLE:
                self._scrfd = SCRFDDetector(num_faces=num_faces)
                self._scrfd_available = True
                logger.info("✅ Hybrid: SCRFD loaded (primary)")
                print("[CONFIG] 🎯 Face Detector: HYBRID (SCRFD primary + MediaPipe fallback)")
            else:
                raise ImportError("insightface not installed")
        except Exception as e:
            logger.warning(f"SCRFD init failed: {e}. Hybrid = MediaPipe only.")
            print(f"[CONFIG] ⚠️ SCRFD not available, Hybrid = MediaPipe only")
        
        # Always init MediaPipe as fallback
        from models.detector import FaceDetector
        self._mediapipe = FaceDetector(mode=mode, num_faces=num_faces)
        logger.info("✅ Hybrid: MediaPipe loaded (fallback)")

        # Stats
        self._scrfd_calls = 0
        self._mediapipe_fallback_calls = 0

    def detect(self, frame_rgb, tracking_state=None):
        """Detect faces using SCRFD primary, MediaPipe fallback.
        
        Returns list of FaceData-compatible objects.
        """
        # Attempt 1: SCRFD (chính xác hơn)
        if self._scrfd_available:
            try:
                faces = self._scrfd.detect(frame_rgb, tracking_state=tracking_state)
                if faces:
                    self._scrfd_calls += 1
                    return faces
            except Exception as e:
                logger.debug(f"SCRFD detect error: {e}")
        
        # Attempt 2: MediaPipe fallback
        faces = self._mediapipe.detect(frame_rgb, tracking_state=tracking_state)
        if self._scrfd_available:
            self._mediapipe_fallback_calls += 1
        return faces

    def get_stats(self):
        """Return detection backend usage stats."""
        total = self._scrfd_calls + self._mediapipe_fallback_calls
        return {
            "scrfd_calls": self._scrfd_calls,
            "mediapipe_fallback_calls": self._mediapipe_fallback_calls,
            "scrfd_ratio": self._scrfd_calls / max(total, 1),
            "total_detections": total,
        }

    def close(self):
        """Cleanup both detectors."""
        if self._scrfd:
            self._scrfd.close()
        if self._mediapipe:
            self._mediapipe.close()
