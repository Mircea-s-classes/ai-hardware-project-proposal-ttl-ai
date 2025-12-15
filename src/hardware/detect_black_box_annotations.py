#!/usr/bin/env python3
"""
Detect black box annotations drawn by user and convert to binary masks.
Handles both bounding boxes and filled rectangles.
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def detect_black_boxes(image, debug=False):
    """
    Detect black rectangular annotations in an image.

    Args:
        image: BGR image with black box annotations
        debug: If True, save debug visualizations

    Returns:
        mask: Binary mask (white=bubble, black=background)
        box_count: Number of boxes detected
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect very dark regions (black boxes)
    # Black boxes will have very low pixel values (< 50)
    black_threshold = 50
    black_mask = (gray < black_threshold).astype(np.uint8) * 255

    # Also detect edges of boxes (stroke rectangles)
    edges = cv2.Canny(gray, 50, 150)

    # Combine both approaches
    combined = cv2.bitwise_or(black_mask, edges)

    # Morphological operations to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.dilate(combined, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out very small detections (noise)
        if w < 20 or h < 20:
            continue

        # Filter out very large detections (likely not a bubble annotation)
        if w > 300 or h > 300:
            continue

        # Check aspect ratio (bubbles should be somewhat round)
        aspect_ratio = float(w) / float(h)
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            continue

        boxes.append((x, y, w, h))

        # Fill this region in the mask
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    if debug:
        debug_img = image.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return mask, len(boxes), debug_img

    return mask, len(boxes)

def process_annotated_frames():
    """
    Process all annotated frames and convert to training data.
    """
    print("=" * 80)
    print(" Converting Black Box Annotations to Training Masks")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parents[2]

    # Look in all directories for annotated frames
    search_dirs = [
        base_dir / "manual_labeling_frames",
        base_dir / "manual_labeling_samples",
        base_dir / "manual_labeling_bubbles3"
    ]

    output_dir = base_dir / "data" / "cnn_manual"
    img_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    debug_dir = output_dir / "debug"

    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    annotated_files = []
    for search_dir in search_dirs:
        if search_dir.exists():
            # Look for any PNG files with black boxes
            annotated_files.extend(list(search_dir.glob("*.png")))

    if not annotated_files:
        print(f"\n✗ No annotated images found")
        return 0

    print(f"\nFound {len(annotated_files)} images to process")
    print(f"Output directory: {output_dir}")

    total_bubbles = 0
    samples_created = 0
    frames_with_bubbles = 0
    frames_without_bubbles = 0

    print("\nProcessing annotations...")
    for img_path in tqdm(annotated_files):
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Detect black boxes
        mask, bubble_count, debug_img = detect_black_boxes(image, debug=True)

        # Save debug visualization
        debug_path = debug_dir / f"debug_{img_path.name}"
        cv2.imwrite(str(debug_path), debug_img)

        if bubble_count > 0:
            total_bubbles += bubble_count
            frames_with_bubbles += 1
        else:
            frames_without_bubbles += 1

        # Create 256x256 crops from this frame
        h, w = image.shape[:2]
        crop_size = 256
        stride = 128

        for y in range(0, h - crop_size + 1, stride):
            for x in range(0, w - crop_size + 1, stride):
                img_crop = image[y:y+crop_size, x:x+crop_size]
                mask_crop = mask[y:y+crop_size, x:x+crop_size]

                # Save ALL crops (including negative samples with no bubbles)
                # This is important for learning what is NOT a bubble
                sample_id = f"{img_path.stem}_y{y}_x{x}"
                cv2.imwrite(str(img_dir / f"{sample_id}.png"), img_crop)
                cv2.imwrite(str(mask_dir / f"{sample_id}.png"), mask_crop)
                samples_created += 1

    print("\n" + "=" * 80)
    print(" Conversion Complete")
    print("=" * 80)
    print(f"Images processed: {len(annotated_files)}")
    print(f"  With bubbles: {frames_with_bubbles}")
    print(f"  Without bubbles (negatives): {frames_without_bubbles}")
    print(f"Total bubbles annotated: {total_bubbles}")
    print(f"Training samples created: {samples_created}")
    print(f"\nSaved to:")
    print(f"  Images: {img_dir}/")
    print(f"  Masks:  {mask_dir}/")
    print(f"  Debug:  {debug_dir}/ (check these to verify detection)")
    print("=" * 80)
    print("\nNext step: Train CNN with class imbalance handling")
    print("  Run: python train_manual_cnn_balanced.py")
    print("=" * 80)

    return samples_created

if __name__ == "__main__":
    process_annotated_frames()
