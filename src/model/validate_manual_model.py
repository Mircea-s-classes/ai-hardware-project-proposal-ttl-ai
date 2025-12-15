#!/usr/bin/env python3
"""
Validate the manually-trained CNN model on both videos.
Compares predictions against user's ground truth expectations.
"""
import torch
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from small_unet import SmallUNet

def load_model():
    """Load the manually-trained model"""
    ckpt_path = Path(__file__).resolve().parents[2] / "data" / "cnn" / "small_unet_manual_trained.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model not found: {ckpt_path}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = SmallUNet().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    print(f"✓ Loaded model from: {ckpt_path}")
    print(f"  Device: {device}")
    print(f"  Training Dice: {state['dice']:.4f}")

    return model, device

def predict_frame(model, device, frame):
    """Run CNN prediction on a single frame"""
    # Prepare input
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        pred = torch.sigmoid(logits).cpu().numpy()[0, 0]

    # Convert to binary mask
    pred_mask = (pred > 0.5).astype(np.uint8) * 255

    return pred_mask

def count_bubbles(mask):
    """Count distinct bubbles in a binary mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by size (bubbles should be at least 100 pixels)
    valid_bubbles = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Minimum bubble size
            valid_bubbles += 1

    return valid_bubbles, contours

def visualize_prediction(frame, pred_mask, bubble_count):
    """Create visualization with predictions overlaid"""
    viz = frame.copy()

    # Overlay prediction in green
    colored_mask = np.zeros_like(frame)
    colored_mask[:, :, 1] = pred_mask  # Green channel
    viz = cv2.addWeighted(viz, 0.7, colored_mask, 0.3, 0)

    # Draw bubble contours
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(viz, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Add text
    cv2.putText(viz, f"Detected: {bubble_count} bubbles", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return viz

def validate_on_video(model, device, video_path, video_name, expected_range, output_dir):
    """Validate model on a video"""
    print(f"\nValidating on {video_name}:")
    print(f"  Expected: {expected_range[0]}-{expected_range[1]} bubbles per frame")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ✗ Could not open video: {video_path}")
        return False

    # Sample frames at 1-second intervals (frames 30, 60, 90, 120, 150)
    test_frames = [30, 60, 90, 120, 150]
    results = []

    for frame_idx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Run prediction
        pred_mask = predict_frame(model, device, frame)
        bubble_count, contours = count_bubbles(pred_mask)

        # Create visualization
        viz = visualize_prediction(frame, pred_mask, bubble_count)

        # Save visualization
        output_path = output_dir / f"{video_name}_frame_{frame_idx:03d}_prediction.png"
        cv2.imwrite(str(output_path), viz)

        # Check if within expected range
        within_range = expected_range[0] <= bubble_count <= expected_range[1]
        status = "✓" if within_range else "✗"

        print(f"  Frame {frame_idx:3d}: {bubble_count} bubbles {status}")
        results.append({
            'frame': frame_idx,
            'count': bubble_count,
            'expected': expected_range,
            'valid': within_range
        })

    cap.release()

    # Summary
    valid_count = sum(r['valid'] for r in results)
    total_count = len(results)
    accuracy = valid_count / total_count if total_count > 0 else 0

    print(f"\n  Summary: {valid_count}/{total_count} frames within expected range ({accuracy*100:.1f}%)")

    return accuracy >= 0.6  # Pass if 60% or more frames are within range

def main():
    print("=" * 80)
    print(" Validating Manually-Trained CNN Model")
    print("=" * 80)

    # Load model
    model, device = load_model()

    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    video_dir = base_dir / "videos"
    output_dir = base_dir / "data" / "cnn_manual" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Validate on both videos
    videos = {
        "AIH_Bubbles": {
            "path": video_dir / "AIH_Bubbles.mp4",
            "expected": (2, 3)  # 2-3 bubbles per frame
        },
        "AIH_Bubbles2": {
            "path": video_dir / "AIH_Bubbles2.mp4",
            "expected": (5, 10)  # 5-10 bubbles per frame
        }
    }

    print("\n" + "=" * 80)
    print(" Validation Results")
    print("=" * 80)

    all_passed = True
    for video_name, config in videos.items():
        passed = validate_on_video(
            model, device,
            config["path"],
            video_name,
            config["expected"],
            output_dir
        )
        all_passed = all_passed and passed

    # Final summary
    print("\n" + "=" * 80)
    if all_passed:
        print(" ✓ VALIDATION PASSED")
        print(" Model predictions are within expected ranges on both videos")
    else:
        print(" ✗ VALIDATION NEEDS REVIEW")
        print(" Some predictions outside expected ranges - check visualizations")

    print(f"\n Visualizations saved to: {output_dir}")
    print("=" * 80)

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
