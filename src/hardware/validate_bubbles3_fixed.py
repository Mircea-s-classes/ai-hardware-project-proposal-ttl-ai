#!/usr/bin/env python3
"""
Validate with FIXED bubble counting - generate sample images
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
    """Group nearby contours into bubble clusters"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 100:
            valid_contours.append(contour)

    return valid_contours

def validate_fixed():
    """Generate sample validation images with fixed counting"""

    base_dir = Path(__file__).parent.parent.parent
    video_path = base_dir / "videos" / "AIH_Bubbles3.mp4"
    model_path = base_dir / "data" / "cnn" / "small_unet_combined_trained.pt"
    output_dir = base_dir / "data" / "validation_bubbles3_fixed"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = SmallUNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    # Sample frames
    frame_indices = [24, 48, 72, 96, 120, 144, 168]

    tile_size = 256
    stride = 128

    print(f"\nProcessing {len(frame_indices)} frames with FIXED counting...")

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.float32)

        # Tile-based inference
        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                tile = frame[y:y+tile_size, x:x+tile_size]
                tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                tile_normalized = tile_rgb.astype(np.float32) / 255.0
                tile_tensor = torch.from_numpy(tile_normalized).permute(2, 0, 1).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(tile_tensor)
                    pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()

                full_mask[y:y+tile_size, x:x+tile_size] = np.maximum(
                    full_mask[y:y+tile_size, x:x+tile_size],
                    pred_mask
                )

        # Binary mask
        binary_mask = (full_mask > 0.5).astype(np.uint8) * 255

        # FIXED: Group bubbles
        bubble_contours = group_contours_into_bubbles(binary_mask, dilation_kernel_size=15)
        num_bubbles = len(bubble_contours)

        # Visualization
        vis = frame.copy()

        # Green mask overlay
        mask_overlay = np.zeros_like(vis)
        mask_overlay[binary_mask > 0] = [0, 255, 0]
        vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)

        # Yellow bounding boxes for bubble clusters
        for contour in bubble_contours:
            x, y, w_box, h_box = cv2.boundingRect(contour)
            cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), (0, 255, 255), 3)

        # Text
        text = f"Frame {frame_idx} | Bubbles: {num_bubbles}"
        cv2.putText(vis, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                   1.2, (0, 255, 255), 3, cv2.LINE_AA)

        # Save
        output_path = output_dir / f"result_frame_{frame_idx:03d}_FIXED.png"
        cv2.imwrite(str(output_path), vis)

        print(f"  Frame {frame_idx}: {num_bubbles} bubbles - saved to {output_path.name}")

    cap.release()

    print(f"\nSaved to: {output_dir}")
    return output_dir

if __name__ == "__main__":
    output = validate_fixed()
    if output:
        import subprocess
        subprocess.run(["open", str(output)])
