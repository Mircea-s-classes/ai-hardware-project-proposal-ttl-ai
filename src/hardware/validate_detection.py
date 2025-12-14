#!/usr/bin/env python3
"""
Validate bubble detection quality across CV, CNN (PyTorch MPS), and CoreML models
"""
import cv2
import sys
from pathlib import Path
import numpy as np

# Add model path
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from bubble_cv_model import BubbleCVModel
from bubble_cnn_model import BubbleCNNModel
from bubble_coreml_model import BubbleCoreMLModel

def test_models():
    print("=" * 80)
    print(" Bubble Detection Validation")
    print("=" * 80)

    # Paths
    video_path = Path(__file__).resolve().parents[2] / "videos" / "AIH_Bubbles.mp4"
    ckpt_path = Path(__file__).resolve().parents[2] / "data" / "cnn" / "small_unet_bubbles.pt"
    mlpackage_path = ckpt_path.parent / "small_unet_bubbles_fp16.mlpackage"

    # Load video
    print(f"\nLoading video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Read test frames (middle of video to avoid blank frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)  # Jump to frame 50
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Could not read test frame")

    print(f"✓ Loaded test frame (frame 50): {frame.shape}")

    # Test 1: CV Model
    print("\n" + "-" * 80)
    print("Testing BubbleCVModel (Classical OpenCV)...")
    print("-" * 80)
    cv_model = BubbleCVModel(min_diam_px=5)
    mask_cv, bubbles_cv = cv_model.predict(frame)

    print(f"✓ CV Model detected: {len(bubbles_cv)} bubbles")
    if len(bubbles_cv) > 0:
        print(f"  Sample bubbles (first 3):")
        for i, b in enumerate(bubbles_cv[:3]):
            print(f"    [{i+1}] x={b['x']}, y={b['y']}, w={b['w']}, h={b['h']}, diam={b['diam_px']:.1f}px")

    # Test 2: CNN Model (PyTorch MPS)
    print("\n" + "-" * 80)
    print("Testing BubbleCNNModel (PyTorch MPS)...")
    print("-" * 80)
    cnn_model = BubbleCNNModel(ckpt_path, min_diam_px=5)
    print(f"  Device: {cnn_model.device}")
    mask_cnn, bubbles_cnn = cnn_model.predict(frame)

    print(f"✓ CNN Model detected: {len(bubbles_cnn)} bubbles")
    if len(bubbles_cnn) > 0:
        print(f"  Sample bubbles (first 3):")
        for i, b in enumerate(bubbles_cnn[:3]):
            print(f"    [{i+1}] x={b['x']}, y={b['y']}, w={b['w']}, h={b['h']}, diam={b['diam_px']:.1f}px")

    # Test 3: CoreML Model
    print("\n" + "-" * 80)
    print("Testing BubbleCoreMLModel (Neural Engine)...")
    print("-" * 80)
    coreml_model = BubbleCoreMLModel(mlpackage_path, min_diam_px=5)
    print(f"  Device: {coreml_model.device}")
    mask_coreml, bubbles_coreml = coreml_model.predict(frame)

    print(f"✓ CoreML Model detected: {len(bubbles_coreml)} bubbles")
    if len(bubbles_coreml) > 0:
        print(f"  Sample bubbles (first 3):")
        for i, b in enumerate(bubbles_coreml[:3]):
            print(f"    [{i+1}] x={b['x']}, y={b['y']}, w={b['w']}, h={b['h']}, diam={b['diam_px']:.1f}px")

    # Compare masks
    print("\n" + "=" * 80)
    print(" Mask Comparison")
    print("=" * 80)
    print(f"CV mask - nonzero pixels: {np.count_nonzero(mask_cv)}")
    print(f"CNN mask - nonzero pixels: {np.count_nonzero(mask_cnn)}")
    print(f"CoreML mask - nonzero pixels: {np.count_nonzero(mask_coreml)}")

    # Calculate overlap between CNN and CoreML
    cnn_binary = (mask_cnn > 0).astype(np.uint8)
    coreml_binary = (mask_coreml > 0).astype(np.uint8)
    intersection = np.logical_and(cnn_binary, coreml_binary).sum()
    union = np.logical_or(cnn_binary, coreml_binary).sum()
    iou = intersection / union if union > 0 else 0

    print(f"\nCNN vs CoreML IoU (should be ~1.0): {iou:.4f}")

    # Summary
    print("\n" + "=" * 80)
    print(" Detection Summary")
    print("=" * 80)
    print(f"CV Model:     {len(bubbles_cv):3d} bubbles")
    print(f"CNN (MPS):    {len(bubbles_cnn):3d} bubbles")
    print(f"CoreML (ANE): {len(bubbles_coreml):3d} bubbles")

    # Validation checks
    print("\n" + "=" * 80)
    print(" Validation Checks")
    print("=" * 80)

    checks_passed = 0
    checks_total = 0

    # Check 1: Models detect bubbles
    checks_total += 1
    if len(bubbles_cv) > 0 and len(bubbles_cnn) > 0 and len(bubbles_coreml) > 0:
        print("✓ All models detect bubbles")
        checks_passed += 1
    else:
        print("✗ Some models detected 0 bubbles - PROBLEM!")

    # Check 2: CNN and CoreML agree (should be nearly identical)
    checks_total += 1
    bubble_diff = abs(len(bubbles_cnn) - len(bubbles_coreml))
    if iou > 0.95:
        print(f"✓ CNN and CoreML masks highly similar (IoU={iou:.4f})")
        checks_passed += 1
    else:
        print(f"✗ CNN and CoreML masks differ (IoU={iou:.4f}) - Check conversion!")

    # Check 3: Bubble counts reasonable
    checks_total += 1
    if 10 <= len(bubbles_cnn) <= 200:  # Reasonable range for bubble video
        print(f"✓ Bubble count in reasonable range ({len(bubbles_cnn)} bubbles)")
        checks_passed += 1
    else:
        print(f"⚠ Unusual bubble count ({len(bubbles_cnn)}) - verify detection quality")

    # Save visualization
    output_dir = Path(__file__).resolve().parents[2] / "videos"

    # Create side-by-side comparison
    vis = np.zeros((frame.shape[0], frame.shape[1] * 3, 3), dtype=np.uint8)

    # CV visualization
    vis_cv = frame.copy()
    for b in bubbles_cv:
        cv2.rectangle(vis_cv, (b['x'], b['y']), (b['x']+b['w'], b['y']+b['h']), (0, 255, 0), 2)
    cv2.putText(vis_cv, f"CV: {len(bubbles_cv)} bubbles", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # CNN visualization
    vis_cnn = frame.copy()
    for b in bubbles_cnn:
        cv2.rectangle(vis_cnn, (b['x'], b['y']), (b['x']+b['w'], b['y']+b['h']), (0, 255, 0), 2)
    cv2.putText(vis_cnn, f"CNN (MPS): {len(bubbles_cnn)} bubbles", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # CoreML visualization
    vis_coreml = frame.copy()
    for b in bubbles_coreml:
        cv2.rectangle(vis_coreml, (b['x'], b['y']), (b['x']+b['w'], b['y']+b['h']), (0, 255, 0), 2)
    cv2.putText(vis_coreml, f"CoreML (ANE): {len(bubbles_coreml)} bubbles", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Combine
    vis[:, :frame.shape[1]] = vis_cv
    vis[:, frame.shape[1]:frame.shape[1]*2] = vis_cnn
    vis[:, frame.shape[1]*2:] = vis_coreml

    comparison_path = output_dir / "detection_comparison.jpg"
    cv2.imwrite(str(comparison_path), vis)
    print(f"\n✓ Saved comparison image: {comparison_path}")

    # Save individual masks
    cv2.imwrite(str(output_dir / "mask_cv.jpg"), mask_cv)
    cv2.imwrite(str(output_dir / "mask_cnn.jpg"), mask_cnn)
    cv2.imwrite(str(output_dir / "mask_coreml.jpg"), mask_coreml)
    print(f"✓ Saved individual masks to {output_dir}/")

    print("\n" + "=" * 80)
    print(f" VALIDATION RESULT: {checks_passed}/{checks_total} checks passed")
    print("=" * 80)

    if checks_passed == checks_total:
        print("✓ ALL CHECKS PASSED - Models working correctly!")
    else:
        print("✗ SOME CHECKS FAILED - Review detection quality!")

    return checks_passed == checks_total

if __name__ == "__main__":
    success = test_models()
    sys.exit(0 if success else 1)
