#!/usr/bin/env python3
"""
Export frames with CV and CNN predictions for manual bubble labeling.
Shows what each model detected so you can provide ground truth labels.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from bubble_cv_model_tuned import BubbleCVModel
from bubble_coreml_model import BubbleCoreMLModel

def export_labeling_samples(video_path, output_dir, frame_indices, mlpackage_path):
    """Export frames with CV and CNN detections for manual labeling"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" Exporting Frames for Manual Bubble Labeling")
    print("=" * 80)

    # Load models
    print("\nLoading models...")
    cv_model = BubbleCVModel()
    cnn_model = BubbleCoreMLModel(mlpackage_path, min_diam_px=10)

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    print(f"\nExporting {len(frame_indices)} frames...")
    print("-" * 80)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Get predictions from both models
        cv_mask, cv_bubbles = cv_model.predict(frame)
        cnn_mask, cnn_bubbles = cnn_model.predict(frame)

        print(f"\nFrame {idx}:")
        print(f"  CV detected: {len(cv_bubbles)} bubbles")
        print(f"  CNN detected: {len(cnn_bubbles)} bubbles")

        # Create comparison visualization
        h, w = frame.shape[:2]

        # Original frame with CV boxes (green)
        frame_cv = frame.copy()
        for bubble in cv_bubbles:
            x, y, bw, bh = bubble['x'], bubble['y'], bubble['w'], bubble['h']
            cv2.rectangle(frame_cv, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(frame_cv, f"CV: {len(cv_bubbles)} bubbles", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Original frame with CNN boxes (blue)
        frame_cnn = frame.copy()
        for bubble in cnn_bubbles:
            x, y, bw, bh = bubble['x'], bubble['y'], bubble['w'], bubble['h']
            cv2.rectangle(frame_cnn, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
        cv2.putText(frame_cnn, f"CNN: {len(cnn_bubbles)} bubbles", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Masks
        cv_mask_color = cv2.cvtColor(cv_mask, cv2.COLOR_GRAY2BGR)
        cv_mask_color[cv_mask > 0] = [0, 255, 0]  # Green

        cnn_mask_color = cv2.cvtColor(cnn_mask, cv2.COLOR_GRAY2BGR)
        cnn_mask_color[cnn_mask > 0] = [255, 0, 0]  # Blue

        # Create 2x2 grid
        top_row = np.hstack([frame, frame_cv])
        bottom_row = np.hstack([cv_mask_color, cnn_mask_color])
        comparison = np.vstack([top_row, bottom_row])

        # Add labels
        label_h = 40
        labels = np.zeros((label_h * 2, w * 2, 3), dtype=np.uint8)
        cv2.putText(labels, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(labels, f"CV Detections ({len(cv_bubbles)})", (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(labels, "CV Mask", (10, label_h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(labels, f"CNN Mask ({len(cnn_bubbles)})", (w + 10, label_h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        final = np.vstack([labels, comparison])

        # Save outputs
        output_path = output_dir / f"frame_{idx:03d}_for_labeling.png"
        cv2.imwrite(str(output_path), final)

        # Also save just the original frame for easier annotation
        orig_path = output_dir / f"frame_{idx:03d}_original.png"
        cv2.imwrite(str(orig_path), frame)

        print(f"  Saved: {output_path.name}")

    cap.release()

    print("\n" + "=" * 80)
    print(" Export Complete")
    print("=" * 80)
    print(f"Files saved to: {output_dir}")
    print("\nFor manual labeling:")
    print("  *_for_labeling.png - Shows CV and CNN predictions")
    print("  *_original.png     - Clean frames for manual annotation")
    print("=" * 80)

if __name__ == "__main__":
    video_path = Path(__file__).resolve().parents[2] / "videos" / "AIH_Bubbles.mp4"
    mlpackage_path = Path(__file__).resolve().parents[2] / "data" / "cnn" / "small_unet_real_fp16.mlpackage"
    output_dir = Path(__file__).resolve().parents[2] / "manual_labeling_samples"

    # Export 10 diverse frames for manual labeling
    frame_indices = [15, 30, 45, 60, 75, 90, 105, 120, 150, 180]

    export_labeling_samples(
        video_path=video_path,
        output_dir=output_dir,
        frame_indices=frame_indices,
        mlpackage_path=mlpackage_path
    )
