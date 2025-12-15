#!/usr/bin/env python3
"""
Create video visualization of bubble detection on AIH_Bubbles3.mp4
Uses trained SmallUNet model to process all frames and generate output video
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import sys

# Add model directory to path
sys.path.append(str(Path(__file__).parent.parent / "model"))
from train_manual_cnn_balanced import SmallUNet

def create_detection_video():
    """Process AIH_Bubbles3.mp4 with trained model and create output video"""

    # Paths
    base_dir = Path(__file__).parent.parent.parent
    video_path = base_dir / "videos" / "AIH_Bubbles3.mp4"
    model_path = base_dir / "data" / "cnn" / "small_unet_combined_trained.pt"
    output_dir = base_dir / "data" / "validation_bubbles3"
    output_video = output_dir / "AIH_Bubbles3_detected.mp4"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {model_path}")
    model = SmallUNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded. Best validation Dice: {checkpoint.get('best_val_dice', 'N/A')}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo: {video_path.name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    if not out.isOpened():
        print("Error: Could not open video writer")
        cap.release()
        return

    # Tile-based inference parameters
    tile_size = 256
    stride = 128

    frame_idx = 0
    total_bubbles = 0

    print("\nProcessing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Initialize full mask
        full_mask = np.zeros((h, w), dtype=np.float32)

        # Process in tiles
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                # Extract tile
                tile = frame[y:y+tile_size, x:x+tile_size]
                tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

                # Normalize
                tile_normalized = tile_rgb.astype(np.float32) / 255.0
                tile_tensor = torch.from_numpy(tile_normalized).permute(2, 0, 1).unsqueeze(0).to(device)

                # Predict
                with torch.no_grad():
                    output = model(tile_tensor)
                    pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()

                # Accumulate (max pooling)
                full_mask[y:y+tile_size, x:x+tile_size] = np.maximum(
                    full_mask[y:y+tile_size, x:x+tile_size],
                    pred_mask
                )

        # Threshold to binary mask
        binary_mask = (full_mask > 0.5).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by size
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= 20:  # Minimum bubble size
                valid_contours.append(contour)

        num_bubbles = len(valid_contours)
        total_bubbles += num_bubbles

        # Create visualization
        vis = frame.copy()

        # Draw mask overlay (green)
        mask_overlay = np.zeros_like(vis)
        mask_overlay[binary_mask > 0] = [0, 255, 0]
        vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)

        # Draw bounding boxes
        for contour in valid_contours:
            x, y, w_box, h_box = cv2.boundingRect(contour)
            cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

        # Add text overlay
        text = f"Frame {frame_idx} | Bubbles: {num_bubbles}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Write frame
        out.write(vis)

        # Progress update
        if frame_idx % 20 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"  Frame {frame_idx}/{total_frames} ({progress:.1f}%) - {num_bubbles} bubbles detected")

        frame_idx += 1

    # Cleanup
    cap.release()
    out.release()

    avg_bubbles = total_bubbles / frame_idx if frame_idx > 0 else 0

    print(f"\n{'='*60}")
    print(f"Video processing complete!")
    print(f"  Processed frames: {frame_idx}")
    print(f"  Total bubbles detected: {total_bubbles}")
    print(f"  Average bubbles per frame: {avg_bubbles:.2f}")
    print(f"  Output saved to: {output_video}")
    print(f"{'='*60}\n")

    return output_video

if __name__ == "__main__":
    output = create_detection_video()
    if output:
        print(f"Opening output directory...")
        import subprocess
        subprocess.run(["open", str(output.parent)])
