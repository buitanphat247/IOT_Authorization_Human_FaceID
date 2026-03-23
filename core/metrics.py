"""
Prometheus Metrics Module for Face Recognition System v5.2.
Exposes key performance indicators for monitoring dashboards.

Metrics:
  - face_detections_total: Counter of face detections
  - face_recognitions_total: Counter of recognition attempts (accepted/rejected)
  - onnx_inference_seconds: Histogram of ONNX inference latency
  - quality_gate_rejections_total: Counter of quality gate rejections by reason
  - faiss_search_seconds: Histogram of FAISS search latency
  - enrollment_total: Counter of enrollments
  - active_users_total: Gauge of total registered users

Usage in app.py:
    from metrics import metrics, metrics_endpoint
    app.route("/metrics")(metrics_endpoint)
"""

import time
from logger import get_logger

logger = get_logger("metrics")

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Metrics disabled. "
                    "Install with: pip install prometheus-client")


class FaceMetrics:
    """Prometheus metrics collector for the face recognition system."""

    def __init__(self):
        self.enabled = PROMETHEUS_AVAILABLE

        if not self.enabled:
            return

        # --- Counters ---
        self.detections = Counter(
            "face_detections_total",
            "Total number of face detections",
            ["status"]  # found, not_found
        )

        self.recognitions = Counter(
            "face_recognitions_total",
            "Total recognition attempts",
            ["result"]  # accepted, rejected, uncertain
        )

        self.quality_rejections = Counter(
            "quality_gate_rejections_total",
            "Quality gate rejections by reason",
            ["reason"]  # Mo, Toi, Sang, Nghieng, NhamMat, Rung, NgSang
        )

        self.enrollments = Counter(
            "face_enrollments_total",
            "Total enrollment operations",
            ["status"]  # success, failed
        )

        # --- Histograms ---
        self.onnx_latency = Histogram(
            "onnx_inference_seconds",
            "ONNX model inference latency",
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )

        self.faiss_latency = Histogram(
            "faiss_search_seconds",
            "FAISS vector search latency",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
        )

        self.recognition_latency = Histogram(
            "recognition_pipeline_seconds",
            "Full recognition pipeline latency (detect + embed + match)",
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )

        # --- Gauges ---
        self.active_users = Gauge(
            "active_users_total",
            "Total number of registered users"
        )

        self.faiss_vectors = Gauge(
            "faiss_vectors_total",
            "Total vectors in FAISS index"
        )

        logger.info("Prometheus metrics initialized")

    # --- Recording helpers ---

    def record_detection(self, found=True):
        if not self.enabled:
            return
        self.detections.labels(status="found" if found else "not_found").inc()

    def record_recognition(self, result="accepted"):
        if not self.enabled:
            return
        self.recognitions.labels(result=result).inc()

    def record_quality_rejection(self, reason):
        if not self.enabled:
            return
        self.quality_rejections.labels(reason=reason).inc()

    def record_enrollment(self, success=True):
        if not self.enabled:
            return
        self.enrollments.labels(status="success" if success else "failed").inc()

    def observe_onnx_latency(self, seconds):
        if not self.enabled:
            return
        self.onnx_latency.observe(seconds)

    def observe_faiss_latency(self, seconds):
        if not self.enabled:
            return
        self.faiss_latency.observe(seconds)

    def observe_recognition_latency(self, seconds):
        if not self.enabled:
            return
        self.recognition_latency.observe(seconds)

    def set_active_users(self, count):
        if not self.enabled:
            return
        self.active_users.set(count)

    def set_faiss_vectors(self, count):
        if not self.enabled:
            return
        self.faiss_vectors.set(count)

    def time(self):
        """Context manager / decorator for timing."""
        return time.time()


# Singleton instance
metrics = FaceMetrics()


def metrics_endpoint():
    """Flask endpoint for /metrics (Prometheus scraping)."""
    if not PROMETHEUS_AVAILABLE:
        return "prometheus_client not installed", 503

    from flask import Response
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
