#!/usr/bin/env python3
"""
Validate retrained model on real bubble data and compare with baseline.
This confirms the model learned real bubble characteristics correctly.
"""
import cv2
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from bubble_cv_model_tuned import BubbleCVModel as BubbleCVModelTuned
from bubble_cnn_model import BubbleCNNModel

def validate_retrained_model():
    print("=" * 80)
    print(" Retrained Model Validation")
    print("=" * 80)

    # Paths
    video_path = Path(__file__).resolve().parents[2] / "videos" / "AIH_Bubbles2.mp4"
    retrained_ckpt = Path(__file__).resolve().parents[2] / "data" / "cnn" / "small_unet_real_trained.pt"

    if not retrained_ckpt.exists():
        print(f"\n✗ Retrained model not found: {retrained_ckpt}")
        print("  Run train_real_cnn.py first!")
        return False

    # Load video
    print(f"\nLoading video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))

    # Test on multiple frames
    test_frames = [30, 50, 80, 120, 150]
    results = {
        "cv_tuned": [],
        "cnn_retrained": []
    }

    # Load models
    print("\nLoading models...")
    cv_model = BubbleCVModelTuned()
    cnn_model = BubbleCNNModel(retrained_ckpt, min_diam_px=10)
    print(f"  CV Model: Tuned baseline")
    print(f"  CNN Model: Retrained on real data (device: {cnn_model.device})")

    print("\n" + "-" * 80)
    print(" Frame-by-Frame Comparison")
    print("-" * 80)

    for frame_idx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # CV baseline
        mask_cv, bubbles_cv = cv_model.predict(frame)

        # Retrained CNN
        mask_cnn, bubbles_cnn = cnn_model.predict(frame)

        results["cv_tuned"].append(len(bubbles_cv))
        results["cnn_retrained"].append(len(bubbles_cnn))

        # Calculate IoU
        cv_binary = (mask_cv > 0).astype(np.uint8)
        cnn_binary = (mask_cnn > 0).astype(np.uint8)
        intersection = np.logical_and(cv_binary, cnn_binary).sum()
        union = np.logical_or(cv_binary, cnn_binary).sum()
        iou = intersection / union if union > 0 else 0

        print(f"Frame {frame_idx:3d}: CV={len(bubbles_cv):2d} bubbles, "
              f"CNN={len(bubbles_cnn):2d} bubbles, IoU={iou:.3f}")

    cap.release()

    print("\n" + "=" * 80)
    print(" Validation Summary")
    print("=" * 80)

    avg_cv = np.mean(results["cv_tuned"])
    avg_cnn = np.mean(results["cnn_retrained"])

    print(f"Average bubbles detected:")
    print(f"  CV Tuned:        {avg_cv:.1f}")
    print(f"  CNN Retrained:   {avg_cnn:.1f}")
    print(f"  Difference:      {abs(avg_cv - avg_cnn):.1f}")

    # Validation checks
    print("\n" + "=" * 80)
    print(" Validation Checks")
    print("=" * 80)

    checks_passed = 0
    checks_total = 0

    # Check 1: CNN detects reasonable number of bubbles
    checks_total += 1
    if 5 <= avg_cnn <= 20:
        print("✓ CNN detects reasonable bubble count (5-20 per frame)")
        checks_passed += 1
    else:
        print(f"✗ CNN bubble count unusual ({avg_cnn:.1f})")

    # Check 2: CNN and CV agree roughly
    checks_total += 1
    if abs(avg_cv - avg_cnn) < 5:
        print("✓ CNN and CV models agree (difference < 5)")
        checks_passed += 1
    else:
        print(f"⚠ CNN and CV differ significantly (diff={abs(avg_cv - avg_cnn):.1f})")

    # Check 3: Not over-predicting (whole frame)
    checks_total += 1
    if all(c < 50 for c in results["cnn_retrained"]):
        print("✓ CNN not over-predicting (all counts < 50)")
        checks_passed += 1
    else:
        print("✗ CNN over-predicting on some frames")

    print("\n" + "=" * 80)
    print(f" RESULT: {checks_passed}/{checks_total} checks passed")
    print("=" * 80)

    if checks_passed == checks_total:
        print("✓ Model validation PASSED! Ready for deployment.")
        return True
    else:
        print("⚠ Some validation checks failed. Review model quality.")
        return False

if __name__ == "__main__":
    success = validate_retrained_model()
    sys.exit(0 if success else 1)
