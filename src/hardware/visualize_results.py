#!/usr/bin/env python3
"""
Visualize bubble detection results showing masks and classifications.
Creates comparison images showing original, mask, and detected bubbles.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from bubble_coreml_model import BubbleCoreMLModel

def visualize_detections(video_path, mlpackage_path, output_dir, frame_indices=[30, 60, 90, 120, 150]):
    """
    Extract frames and visualize detection results.

    Args:
        video_path: Path to input video
        mlpackage_path: Path to CoreML model
        output_dir: Where to save visualization images
        frame_indices: Which frames to visualize
    """
    print("=" * 80)
    print(" Bubble Detection Visualization")
    print("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading CoreML model: {mlpackage_path}")
    model = BubbleCoreMLModel(mlpackage_path, min_diam_px=10)

    # Open video
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✓ Video loaded: {total_frames} frames\n")

    print("=" * 80)
    print(" Extracting and Processing Frames")
    print("=" * 80)

    for idx in frame_indices:
        if idx >= total_frames:
            print(f"⚠ Frame {idx} out of range, skipping")
            continue

        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            print(f"⚠ Could not read frame {idx}")
            continue

        # Run detection
        mask, bubbles = model.predict(frame)

        print(f"\nFrame {idx}:")
        print(f"  Bubbles detected: {len(bubbles)}")

        # Create visualizations
        h, w = frame.shape[:2]

        # 1. Original frame
        frame_orig = frame.copy()

        # 2. Mask visualization (grayscale to color)
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_color[mask > 0] = [0, 255, 0]  # Green for bubbles

        # 3. Overlay mask on frame (semi-transparent)
        overlay = frame.copy()
        overlay[mask > 0] = cv2.addWeighted(
            overlay[mask > 0], 0.6,
            mask_color[mask > 0], 0.4,
            0
        )

        # 4. Draw bounding boxes and labels
        frame_boxes = frame.copy()
        for i, bubble in enumerate(bubbles):
            x, y, bw, bh = bubble['x'], bubble['y'], bubble['w'], bubble['h']
            diam = bubble['diam_px']

            # Draw bounding box (green)
            cv2.rectangle(frame_boxes, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            # Draw label
            label = f"#{i+1}: {diam:.1f}px"
            cv2.putText(frame_boxes, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add detection count
        count_text = f"Bubbles: {len(bubbles)}"
        cv2.putText(frame_boxes, count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 5. Create comparison grid (2x2)
        top_row = np.hstack([frame_orig, mask_color])
        bottom_row = np.hstack([overlay, frame_boxes])
        comparison = np.vstack([top_row, bottom_row])

        # Add labels to each quadrant
        label_h, label_w = 40, w
        labels = np.zeros((label_h * 2, label_w * 2, 3), dtype=np.uint8)

        cv2.putText(labels, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(labels, "Mask (Green=Bubble)", (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(labels, "Overlay", (10, label_h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(labels, f"Detected ({len(bubbles)} bubbles)", (w + 10, label_h + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        final = np.vstack([labels, comparison])

        # Save individual outputs
        output_path = output_dir / f"frame_{idx:03d}_comparison.png"
        cv2.imwrite(str(output_path), final)
        print(f"  ✓ Saved: {output_path.name}")

        # Also save just the boxes version for quick view
        boxes_path = output_dir / f"frame_{idx:03d}_boxes.png"
        cv2.imwrite(str(boxes_path), frame_boxes)

        # Save just the mask
        mask_path = output_dir / f"frame_{idx:03d}_mask.png"
        cv2.imwrite(str(mask_path), mask_color)

        # Print bubble details
        if len(bubbles) > 0:
            print(f"  Bubble details:")
            for i, bubble in enumerate(bubbles):
                print(f"    Bubble {i+1}: pos=({bubble['x']}, {bubble['y']}), "
                      f"size={bubble['w']}x{bubble['h']}, diam={bubble['diam_px']:.1f}px")

    cap.release()

    print("\n" + "=" * 80)
    print(" Visualization Complete")
    print("=" * 80)
    print(f"Images saved to: {output_dir}")
    print("\nFiles created:")
    print("  *_comparison.png - 2x2 grid showing all visualizations")
    print("  *_boxes.png      - Original frame with bounding boxes")
    print("  *_mask.png       - Binary mask visualization")
    print("=" * 80)

if __name__ == "__main__":
    video_path = Path(__file__).resolve().parents[2] / "videos" / "AIH_Bubbles.mp4"
    mlpackage_path = Path(__file__).resolve().parents[2] / "data" / "cnn" / "small_unet_real_fp16.mlpackage"
    output_dir = Path(__file__).resolve().parents[2] / "visualization_output"

    visualize_detections(
        video_path=video_path,
        mlpackage_path=mlpackage_path,
        output_dir=output_dir,
        frame_indices=[30, 60, 90, 120, 150]  # 5 sample frames
    )
