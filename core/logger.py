"""
Centralized Logging Module for Face Recognition System v5.2
Replaces all print() calls with structured Python logging.
Supports: File rotation, Console output, JSON format option.
"""

import os
import logging
import logging.handlers
from config import ROOT_DIR

# === LOG CONFIG ===
LOG_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "face_system.log")
LOG_LEVEL = logging.INFO
LOG_MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
LOG_BACKUP_COUNT = 5               # Keep 5 rotated files


def get_logger(name: str) -> logging.Logger:
    """Get a named logger with file + console handlers.
    
    Usage:
        from logger import get_logger
        logger = get_logger(__name__)
        logger.info("System started")
        logger.warning("Low quality face detected")
        logger.error("FAISS index corrupted", exc_info=True)
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers if already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False
    
    # --- Format ---
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # --- Console Handler (stdout) ---
    console = logging.StreamHandler()
    console.setLevel(LOG_LEVEL)
    console.setFormatter(fmt)
    logger.addHandler(console)
    
    # --- Rotating File Handler ---
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8"
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    
    return logger
