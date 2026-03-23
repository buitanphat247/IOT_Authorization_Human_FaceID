"""
Quality Score Weight Optimizer - Data-driven weight tuning.
Uses Logistic Regression to find optimal weights for the quality score formula.

Current formula (detector.py):
    score = blur*0.30 + bright*0.20 + angle*0.20 + eye*0.15 + size*0.15

This script:
    1. Collects quality features from test images (good + bad)
    2. Trains Logistic Regression to classify good/bad
    3. Extracts learned weights as optimal quality score formula
    4. Prints updated config values

Usage:
    python scripts/optimize_weights.py --data-dir data/quality_test
    
Data structure:
    data/quality_test/
        good/   (clear, well-lit, frontal face images)
        bad/    (blurry, dark, tilted, eyes closed images)
"""

import os
import sys
import argparse
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "core"))


def extract_features(img, face):
    """Extract 5 quality features from a face detection."""
    import cv2
    from config import (MIN_FACE_SIZE, BLUR_THRESH, MIN_BRIGHT, MAX_BRIGHT, MAX_TILT)

    x, y, w, h = face.bbox
    fh, fw = img.shape[:2]
    roi = img[max(0, y):min(fh, y + h), max(0, x):min(fw, x + w)]
    
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Feature 1: Blur score (Laplacian variance, normalized)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(blur / 200.0, 1.0)

    # Feature 2: Brightness score
    bright = np.mean(gray)
    bright_score = 1.0 - abs(bright - 130) / 130.0

    # Feature 3: Angle score
    angle = abs(np.degrees(np.arctan2(
        face.lm5[1][1] - face.lm5[0][1],
        face.lm5[1][0] - face.lm5[0][0]
    )))
    angle_score = 1.0 - angle / 45.0

    # Feature 4: Eye openness score
    _, _, avg_ear = face.eye_openness()
    eye_score = min(avg_ear / 0.05, 1.0)

    # Feature 5: Face size score
    face_size_score = min(w * h / (160 * 160), 1.0)

    return np.array([blur_score, bright_score, max(0, angle_score), eye_score, face_size_score])


def collect_data(data_dir):
    """Collect features from good/ and bad/ subdirectories."""
    import cv2
    from detector import FaceDetector

    detector = FaceDetector(mode="image", num_faces=1)
    
    features = []
    labels = []
    
    for label, subdir in [(1, "good"), (0, "bad")]:
        dir_path = os.path.join(data_dir, subdir)
        if not os.path.exists(dir_path):
            print(f"[WARNING] Directory not found: {dir_path}")
            continue

        files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  [{subdir}] Found {len(files)} images")

        for fname in files:
            img = cv2.imread(os.path.join(dir_path, fname))
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect(rgb)

            if len(faces) != 1:
                continue

            feat = extract_features(img, faces[0])
            if feat is not None:
                features.append(feat)
                labels.append(label)

    detector.close()
    return np.array(features), np.array(labels)


def optimize_weights(features, labels):
    """Train Logistic Regression and extract weights."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("[ERROR] scikit-learn not installed. Install with:")
        print("  pip install scikit-learn")
        return None

    print(f"\n[TRAINING] {len(features)} samples ({sum(labels)} good, {len(labels) - sum(labels)} bad)")

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = labels

    # Train with cross-validation
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"  Cross-validation accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

    # Final fit
    model.fit(X, y)
    
    # Extract and normalize weights
    raw_weights = np.abs(model.coef_[0])
    normalized_weights = raw_weights / raw_weights.sum()

    feature_names = ["blur", "brightness", "angle", "eye_openness", "face_size"]
    
    print(f"\n{'='*50}")
    print(f"  OPTIMIZED QUALITY SCORE WEIGHTS:")
    print(f"{'='*50}")
    for name, w in zip(feature_names, normalized_weights):
        print(f"    {name:15s}: {w:.2f} ({w*100:.0f}%)")
    
    print(f"\n  Update detector.py line ~259 with:")
    print(f"    score = (blur_score * {normalized_weights[0]:.2f}")
    print(f"             + bright_score * {normalized_weights[1]:.2f}")
    print(f"             + angle_score * {normalized_weights[2]:.2f}")
    print(f"             + eye_score * {normalized_weights[3]:.2f}")
    print(f"             + face_size_score * {normalized_weights[4]:.2f})")
    print(f"{'='*50}")

    return normalized_weights


def generate_synthetic_data():
    """Generate synthetic test data when no real data available."""
    print("\n[INFO] Generating synthetic data for demonstration...")
    
    np.random.seed(42)
    n_good = 200
    n_bad = 200
    
    # Good images: high blur_score, good brightness, low angle, open eyes, decent size
    good_features = np.column_stack([
        np.random.uniform(0.5, 1.0, n_good),   # blur
        np.random.uniform(0.6, 1.0, n_good),   # brightness
        np.random.uniform(0.7, 1.0, n_good),   # angle
        np.random.uniform(0.5, 1.0, n_good),   # eye
        np.random.uniform(0.4, 1.0, n_good),   # size
    ])
    
    # Bad images: low quality in at least one dimension
    bad_features = np.column_stack([
        np.random.uniform(0.0, 0.5, n_bad),    # blur (main issue)
        np.random.uniform(0.1, 0.7, n_bad),    # brightness
        np.random.uniform(0.0, 0.6, n_bad),    # angle
        np.random.uniform(0.0, 0.5, n_bad),    # eye
        np.random.uniform(0.1, 0.6, n_bad),    # size
    ])
    
    features = np.vstack([good_features, bad_features])
    labels = np.array([1]*n_good + [0]*n_bad)
    
    return features, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize quality score weights")
    parser.add_argument("--data-dir", default=os.path.join(ROOT_DIR, "data", "quality_test"),
                        help="Directory with good/ and bad/ subdirectories")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data for demonstration")
    args = parser.parse_args()

    if args.synthetic:
        features, labels = generate_synthetic_data()
    else:
        if not os.path.exists(args.data_dir):
            print(f"[WARNING] Data directory not found: {args.data_dir}")
            print(f"  Create this structure:")
            print(f"    {args.data_dir}/good/  (clear face images)")
            print(f"    {args.data_dir}/bad/   (blurry/dark/tilted images)")
            print(f"\n  Or use --synthetic for demonstration")
            sys.exit(1)
        features, labels = collect_data(args.data_dir)

    if len(features) < 20:
        print(f"[ERROR] Not enough data ({len(features)} samples). Need at least 20.")
        sys.exit(1)

    weights = optimize_weights(features, labels)
