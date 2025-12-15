#!/usr/bin/env python3
"""
Tune CV model parameters to reduce false positives and improve bubble detection accuracy.
This creates the baseline for generating training labels from real data.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

class BubbleCVModelTuned:
    """
    Improved CV model with tuned parameters for real bubble detection.
    Reduces false positives by:
    - Better thresholding
    - Stricter morphology
    - Filtering by shape/size characteristics
    """
    def __init__(self, min_diam_px=10, max_diam_px=300, min_circularity=0.3):
        self.min_diam_px = min_diam_px
        self.max_diam_px = max_diam_px
        self.min_circularity = min_circularity

    def predict(self, frame_bgr):
        """
        Improved bubble detection with reduced false positives
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu threshold
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Stronger morphology to clean up noise
        kernel = np.ones((5, 5), np.uint8)
        th_clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
        th_clean = cv2.morphologyEx(th_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            th_clean, connectivity=8
        )

        bubbles = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            diam_px = max(w, h)

            # Filter by size
            if diam_px < self.min_diam_px or diam_px > self.max_diam_px:
                continue

            # Filter by aspect ratio (bubbles should be roughly circular)
            aspect_ratio = float(w) / float(h) if h > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

            # Filter by circularity (4*pi*area / perimeter^2)
            mask_i = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < self.min_circularity:
                        continue

            # Filter by position (exclude edge artifacts)
            margin = 5
            h_frame, w_frame = frame_bgr.shape[:2]
            if x < margin or y < margin or (x + w) > (w_frame - margin) or (y + h) > (h_frame - margin):
                continue

            bubbles.append({
                "x": int(x), "y": int(y),
                "w": int(w), "h": int(h),
                "diam_px": float(diam_px),
            })

        return th_clean, bubbles

def interactive_tuning():
    """
    Interactive tool to tune parameters on sample frames
    """
    print("=" * 80)
    print(" CV Model Parameter Tuning")
    print("=" * 80)

    video_path = Path(__file__).resolve().parents[2] / "videos" / "AIH_Bubbles.mp4"

    # Load sample frames
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_indices = [30, 50, 80, 120, 150]  # Sample frames

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    cap.release()

    print(f"\nLoaded {len(frames)} sample frames for tuning")

    # Test different parameter sets
    param_sets = [
        {"min_diam_px": 10, "max_diam_px": 300, "min_circularity": 0.3, "name": "Default"},
        {"min_diam_px": 15, "max_diam_px": 250, "min_circularity": 0.4, "name": "Stricter"},
        {"min_diam_px": 20, "max_diam_px": 200, "min_circularity": 0.5, "name": "Very Strict"},
        {"min_diam_px": 8, "max_diam_px": 350, "min_circularity": 0.25, "name": "Lenient"},
    ]

    results = []

    for params in param_sets:
        model = BubbleCVModelTuned(**{k: v for k, v in params.items() if k != "name"})

        bubble_counts = []
        for idx, frame in frames:
            mask, bubbles = model.predict(frame)
            bubble_counts.append(len(bubbles))

        avg_count = np.mean(bubble_counts)
        results.append({
            "name": params["name"],
            "params": params,
            "avg_bubbles": avg_count,
            "counts": bubble_counts
        })

        print(f"\n{params['name']} parameters:")
        print(f"  min_diam={params['min_diam_px']}, max_diam={params['max_diam_px']}, min_circ={params['min_circularity']}")
        print(f"  Detected bubbles per frame: {bubble_counts}")
        print(f"  Average: {avg_count:.1f} bubbles")

    print("\n" + "=" * 80)
    print(" Recommendation")
    print("=" * 80)

    # Find params with reasonable count (expecting 3-20 bubbles per frame for real data)
    best = min(results, key=lambda x: abs(x["avg_bubbles"] - 10))

    print(f"\nBest parameter set: {best['name']}")
    print(f"  Average bubbles: {best['avg_bubbles']:.1f}")
    print(f"  Parameters: {best['params']}")
    print("\nSaving best parameters...")

    # Save best model
    output_path = Path(__file__).parent / "bubble_cv_model_tuned.py"
    with open(output_path, 'w') as f:
        f.write(f'''# Auto-generated tuned CV model
from pathlib import Path
import cv2
import numpy as np

class BubbleCVModel:
    """Tuned CV model for real bubble detection"""
    def __init__(self, min_diam_px={best['params']['min_diam_px']}, max_diam_px={best['params']['max_diam_px']}, min_circularity={best['params']['min_circularity']}):
        self.min_diam_px = min_diam_px
        self.max_diam_px = max_diam_px
        self.min_circularity = min_circularity

    def predict(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        th_clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
        th_clean = cv2.morphologyEx(th_clean, cv2.MORPH_CLOSE, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th_clean, connectivity=8)

        bubbles = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            diam_px = max(w, h)

            if diam_px < self.min_diam_px or diam_px > self.max_diam_px:
                continue

            aspect_ratio = float(w) / float(h) if h > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue

            mask_i = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < self.min_circularity:
                        continue

            h_frame, w_frame = frame_bgr.shape[:2]
            margin = 5
            if x < margin or y < margin or (x + w) > (w_frame - margin) or (y + h) > (h_frame - margin):
                continue

            bubbles.append({{
                "x": int(x), "y": int(y),
                "w": int(w), "h": int(h),
                "diam_px": float(diam_px),
            }})

        return th_clean, bubbles
''')

    print(f"✓ Saved tuned model to: {output_path}")
    print("\nNext steps:")
    print("  1. Use this tuned model to generate training labels")
    print("  2. Extract frames from AIH_Bubbles.mp4")
    print("  3. Retrain CNN on real data")

    return best

if __name__ == "__main__":
    interactive_tuning()
