#!/usr/bin/env python3
"""
Validate trained CNN model on AIH_Bubbles3.mp4 video.
"""
import torch
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from small_unet import SmallUNet


def validate_on_bubbles3():
    print("=" * 80)
    print(" Validating Trained Model on AIH_Bubbles3.mp4")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base_dir = Path(__file__).resolve().parents[2]

    video_path = base_dir / "videos" / "AIH_Bubbles3.mp4"
    model_path = base_dir / "data" / "cnn" / "small_unet_combined_trained.pt"
    output_dir = base_dir / "data" / "validation_bubbles3"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        return

    # Load model
    print(f"\nLoading model from: {model_path.name}")
    model = SmallUNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded (Validation Dice: {checkpoint.get('dice', 0):.4f})")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nVideo: {video_path.name}")
    print(f"Total frames: {total_frames}, FPS: {fps:.1f}")

    # Process sample frames
    frame_indices = [0, 24, 48, 72, 96, 120, 144, 168]
    print(f"\nProcessing {len(frame_indices)} frames...")

    results = []

    for frame_idx in frame_indices:
        if frame_idx >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.float32)

        # Process in tiles
        tile_size = 256
        stride = 128

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
                    full_mask[y:y+tile_size, x:x+tile_size], pred_mask
                )

        # Threshold and find bubbles
        binary_mask = (full_mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bubbles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:
                continue
            x, y, w_box, h_box = cv2.boundingRect(contour)
            bubbles.append({'bbox': (x, y, w_box, h_box), 'area': area})

        # Visualize
        vis = frame.copy()
        mask_overlay = np.zeros_like(frame)
        mask_overlay[:, :, 1] = binary_mask
        vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)

        for bubble in bubbles:
            x, y, w_box, h_box = bubble['bbox']
            cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

        info_text = f"Frame {frame_idx} | Bubbles: {len(bubbles)}"
        cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        output_path = output_dir / f"result_frame_{frame_idx:03d}.png"
        cv2.imwrite(str(output_path), vis)

        results.append({'frame': frame_idx, 'bubbles': len(bubbles), 'output': output_path})
        print(f"  Frame {frame_idx:3d}: {len(bubbles)} bubbles")

    cap.release()

    print("\n" + "=" * 80)
    print(" Validation Complete")
    print("=" * 80)
    total_bubbles = sum(r['bubbles'] for r in results)
    avg_bubbles = total_bubbles / len(results) if results else 0
    print(f"Frames: {len(results)}, Total bubbles: {total_bubbles}, Avg: {avg_bubbles:.1f}")
    print(f"Results: {output_dir}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    validate_on_bubbles3()
