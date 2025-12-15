#!/usr/bin/env python3
"""
Generate training dataset from AIH_Bubbles.mp4 using tuned CV model.
This creates the real-data training set for retraining the CNN.
"""
import cv2
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Add hardware path for tuned CV model
sys.path.insert(0, str(Path(__file__).parent.parent / "hardware"))
from bubble_cv_model_tuned import BubbleCVModel

def extract_training_data(
    video_path,
    output_dir,
    frame_stride=5,  # Sample every Nth frame
    crop_size=256,
    min_bubbles_per_frame=2  # Only save frames with enough bubbles
):
    """
    Extract training data from real bubble video

    Args:
        video_path: Path to AIH_Bubbles.mp4
        output_dir: Where to save images/masks
        frame_stride: Sample every Nth frame (5 = 20% of frames)
        crop_size: Size of training crops (256x256)
        min_bubbles_per_frame: Minimum bubbles to consider frame valid
    """
    print("=" * 80)
    print(" Real Data Training Set Generation")
    print("=" * 80)

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    # Create output directories
    img_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nInput video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Frame stride: {frame_stride} (sampling {100//frame_stride}% of frames)")
    print(f"Crop size: {crop_size}x{crop_size}")

    # Load tuned CV model
    print("\nLoading tuned CV model...")
    model = BubbleCVModel()

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✓ Video: {total_frames} frames @ {fps} FPS")

    # Process frames
    frame_idx = 0
    saved_count = 0
    skipped_count = 0

    pbar = tqdm(total=total_frames // frame_stride, desc="Generating training data")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every Nth frame
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        # Generate mask using tuned CV model
        mask, bubbles = model.predict(frame)

        # Skip frames with too few bubbles
        if len(bubbles) < min_bubbles_per_frame:
            skipped_count += 1
            frame_idx += 1
            pbar.update(1)
            continue

        # Generate multiple 256x256 crops from this frame
        h, w = frame.shape[:2]

        # Random crops with augmentation
        crops_per_frame = 3  # Generate 3 random crops per valid frame

        for crop_idx in range(crops_per_frame):
            # Random crop position (ensure we don't go out of bounds)
            if w > crop_size and h > crop_size:
                x = np.random.randint(0, w - crop_size)
                y = np.random.randint(0, h - crop_size)
            else:
                # If frame smaller than crop, resize first
                scale = max(crop_size / w, crop_size / h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                x = np.random.randint(0, new_w - crop_size) if new_w > crop_size else 0
                y = np.random.randint(0, new_h - crop_size) if new_h > crop_size else 0

            # Extract crop
            img_crop = frame[y:y+crop_size, x:x+crop_size]
            mask_crop = mask[y:y+crop_size, x:x+crop_size]

            # Skip if crop has no bubbles (all black mask)
            if np.count_nonzero(mask_crop) < 100:  # At least 100 pixels
                continue

            # Data augmentation (50% chance each)
            if np.random.rand() < 0.5:
                # Horizontal flip
                img_crop = cv2.flip(img_crop, 1)
                mask_crop = cv2.flip(mask_crop, 1)

            if np.random.rand() < 0.3:
                # Small rotation (-10 to +10 degrees)
                angle = np.random.uniform(-10, 10)
                M = cv2.getRotationMatrix2D((crop_size//2, crop_size//2), angle, 1.0)
                img_crop = cv2.warpAffine(img_crop, M, (crop_size, crop_size))
                mask_crop = cv2.warpAffine(mask_crop, M, (crop_size, crop_size))

            # Save
            sample_id = f"aih_f{frame_idx:05d}_c{crop_idx}"
            cv2.imwrite(str(img_dir / f"{sample_id}.png"), img_crop)
            cv2.imwrite(str(mask_dir / f"{sample_id}.png"), mask_crop)
            saved_count += 1

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    print("\n" + "=" * 80)
    print(" Dataset Generation Complete")
    print("=" * 80)
    print(f"Total frames processed: {frame_idx}")
    print(f"Frames used: {frame_idx // frame_stride - skipped_count}")
    print(f"Frames skipped (too few bubbles): {skipped_count}")
    print(f"Training samples generated: {saved_count}")
    print(f"\nSaved to:")
    print(f"  Images: {img_dir}/ ({len(list(img_dir.glob('*.png')))} files)")
    print(f"  Masks:  {mask_dir}/ ({len(list(mask_dir.glob('*.png')))} files)")
    print("=" * 80)

    return saved_count

if __name__ == "__main__":
    video_path = Path(__file__).resolve().parents[2] / "videos" / "AIH_Bubbles.mp4"
    output_dir = Path(__file__).resolve().parents[2] / "data" / "cnn_real"

    count = extract_training_data(
        video_path=video_path,
        output_dir=output_dir,
        frame_stride=3,  # Sample every 3rd frame (33% of video)
        crop_size=256,
        min_bubbles_per_frame=3  # Need at least 3 bubbles
    )

    print(f"\nNext step: Retrain CNN on {count} real bubble samples")
    print("  Run: python train_real_cnn.py")
