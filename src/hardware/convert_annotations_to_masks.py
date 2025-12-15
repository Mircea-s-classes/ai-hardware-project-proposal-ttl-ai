#!/usr/bin/env python3
"""
Convert manually annotated frames (with red circles) into binary masks.
Detects red circles and creates ground truth masks for CNN training.
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def detect_red_circles(annotated_image):
    """
    Detect red circles drawn by user and create binary mask.

    Returns:
        Binary mask where white=bubble, black=background
    """
    h, w = annotated_image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Convert to HSV for better red detection
    hsv = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2HSV)

    # Red color range in HSV (two ranges because red wraps around)
    # Lower red (0-10)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    # Upper red (170-180)
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Find contours of red circles
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, fill the interior to create bubble mask
    for contour in contours:
        # Fill the contour
        cv2.drawContours(mask, [contour], -1, 255, -1)

    return mask, len(contours)

def convert_annotations_to_training_data():
    print("=" * 80)
    print(" Converting Manual Annotations to Training Masks")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parents[2]
    input_dir = base_dir / "manual_labeling_frames"
    output_dir = base_dir / "data" / "cnn_manual"

    # Create output directories
    img_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Find all annotated images
    annotated_files = list(input_dir.glob("*_annotated.png"))

    if not annotated_files:
        print(f"\n✗ No annotated images found in {input_dir}")
        print("  Please annotate frames first!")
        print("  Files should be named: *_annotated.png")
        return 0

    print(f"\nFound {len(annotated_files)} annotated images")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    total_bubbles = 0
    samples_created = 0

    print("\nProcessing annotations...")
    for annotated_path in tqdm(annotated_files):
        # Load annotated image
        annotated = cv2.imread(str(annotated_path))

        if annotated is None:
            print(f"  ✗ Failed to load: {annotated_path.name}")
            continue

        # Find corresponding original frame
        original_name = annotated_path.name.replace("_annotated", "")
        original_path = input_dir / original_name

        if not original_path.exists():
            print(f"  ✗ Original frame not found: {original_name}")
            continue

        # Load original frame
        original = cv2.imread(str(original_path))

        # Detect red circles and create mask
        mask, bubble_count = detect_red_circles(annotated)
        total_bubbles += bubble_count

        # Create 256x256 crops from this frame
        h, w = original.shape[:2]
        crop_size = 256

        # Sliding window with stride
        stride = 128
        crop_idx = 0

        for y in range(0, h - crop_size + 1, stride):
            for x in range(0, w - crop_size + 1, stride):
                # Extract crop
                img_crop = original[y:y+crop_size, x:x+crop_size]
                mask_crop = mask[y:y+crop_size, x:x+crop_size]

                # Only save if crop contains bubbles
                if np.sum(mask_crop > 0) < 10:  # At least 10 bubble pixels
                    continue

                # Save crop
                sample_id = f"{annotated_path.stem}_crop{crop_idx:02d}"
                cv2.imwrite(str(img_dir / f"{sample_id}.png"), img_crop)
                cv2.imwrite(str(mask_dir / f"{sample_id}.png"), mask_crop)
                samples_created += 1
                crop_idx += 1

    print("\n" + "=" * 80)
    print(" Conversion Complete")
    print("=" * 80)
    print(f"Annotated frames processed: {len(annotated_files)}")
    print(f"Total bubbles annotated: {total_bubbles}")
    print(f"Training samples created: {samples_created}")
    print(f"\nSaved to:")
    print(f"  Images: {img_dir}/ ({len(list(img_dir.glob('*.png')))} files)")
    print(f"  Masks:  {mask_dir}/ ({len(list(mask_dir.glob('*.png')))} files)")
    print("=" * 80)
    print("\nNext step: Retrain CNN on ground truth data")
    print("  Run: python train_manual_cnn.py")
    print("=" * 80)

    return samples_created

if __name__ == "__main__":
    convert_annotations_to_training_data()
