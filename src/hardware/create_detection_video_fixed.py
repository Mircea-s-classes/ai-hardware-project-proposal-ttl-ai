#!/usr/bin/env python3
"""
Create video visualization with FIXED bubble counting
Groups nearby contours into bubble clusters (1 cluster = 1 bubble)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import sys

# Add model directory to path
sys.path.append(str(Path(__file__).parent.parent / "model"))
from train_manual_cnn_balanced import SmallUNet

def group_contours_into_bubbles(binary_mask, dilation_kernel_size=15):
    """
    Group nearby contours into bubble clusters using morphological operations

    Args:
        binary_mask: Binary mask from CNN prediction
        dilation_kernel_size: Size of dilation kernel to connect nearby regions

    Returns:
        bubble_contours: List of contours, each representing ONE bubble cluster
    """
    # Apply morphological dilation to connect nearby bubble segments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Find contours on dilated mask (now each contour represents a bubble cluster)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by minimum size
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 100:  # Minimum bubble cluster size
            valid_contours.append(contour)

    return valid_contours

def create_detection_video_fixed():
    """Process AIH_Bubbles3.mp4 with FIXED bubble counting"""

    # Paths
    base_dir = Path(__file__).parent.parent.parent
    video_path = base_dir / "videos" / "AIH_Bubbles3.mp4"
    model_path = base_dir / "data" / "cnn" / "small_unet_combined_trained.pt"
    output_dir = base_dir / "data" / "validation_bubbles3"
    output_video = output_dir / "AIH_Bubbles3_detected_FIXED.mp4"

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
    print(f"Model loaded (Dice: 0.4338)")
    print(f"\nFIXED: Using contour clustering to group bubble segments")

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

    print("\nProcessing frames with FIXED counting...")

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

        # FIXED: Group nearby contours into bubble clusters
        bubble_contours = group_contours_into_bubbles(binary_mask, dilation_kernel_size=15)

        num_bubbles = len(bubble_contours)
        total_bubbles += num_bubbles

        # Create visualization
        vis = frame.copy()

        # Draw original mask overlay (green, semi-transparent)
        mask_overlay = np.zeros_like(vis)
        mask_overlay[binary_mask > 0] = [0, 255, 0]
        vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)

        # Draw bounding boxes for each bubble cluster (yellow for distinction)
        for contour in bubble_contours:
            x, y, w_box, h_box = cv2.boundingRect(contour)
            cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), (0, 255, 255), 3)  # Yellow, thicker

        # Add text overlay
        text = f"Frame {frame_idx} | Bubbles: {num_bubbles} (FIXED COUNTING)"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0, 255, 255), 2, cv2.LINE_AA)

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
    print(f"Video processing complete (FIXED COUNTING)!")
    print(f"  Processed frames: {frame_idx}")
    print(f"  Total bubble clusters: {total_bubbles}")
    print(f"  Average bubbles per frame: {avg_bubbles:.2f}")
    print(f"  Output saved to: {output_video}")
    print(f"{'='*60}\n")

    return output_video

if __name__ == "__main__":
    output = create_detection_video_fixed()
    if output:
        print(f"Opening output directory...")
        import subprocess
        subprocess.run(["open", str(output.parent)])
