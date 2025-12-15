#!/usr/bin/env python3
"""
Advanced bubble detection for AIH_Bubbles3.mp4 with black background.
Leverages high contrast to produce superior masks for CNN training.
"""
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def detect_bubbles_black_bg(frame, debug=False):
    """
    Detect bubbles in high-contrast black background video.

    Strategy:
    1. Isolate syringe using brightness thresholding
    2. Detect bright regions within syringe (bubbles + liquid)
    3. Use circularity and reflectance to identify bubbles
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 1: Isolate syringe region (anything brighter than black background)
    # Black background is typically < 30, syringe is > 50
    _, syringe_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Clean up syringe mask
    kernel = np.ones((5, 5), np.uint8)
    syringe_mask = cv2.morphologyEx(syringe_mask, cv2.MORPH_CLOSE, kernel)
    syringe_mask = cv2.morphologyEx(syringe_mask, cv2.MORPH_OPEN, kernel)

    # Step 2: Within syringe, detect very bright regions (bubbles)
    # Bubbles are highly reflective: typically > 150 brightness
    _, bright_regions = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # Combine with syringe mask to filter out background
    bubble_candidates = cv2.bitwise_and(bright_regions, syringe_mask)

    # Step 3: Filter by bubble characteristics
    contours, _ = cv2.findContours(bubble_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by size (bubbles are typically 50-5000 pixels)
        if area < 50 or area > 5000:
            continue

        # Check circularity (bubbles are round)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Bubbles should be somewhat circular (> 0.3)
        if circularity < 0.3:
            continue

        # Check aspect ratio (should be close to 1.0 for circles)
        x, y, w_box, h_box = cv2.boundingRect(contour)
        aspect_ratio = float(w_box) / float(h_box) if h_box > 0 else 0

        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            continue

        bubbles.append({
            'contour': contour,
            'area': area,
            'circularity': circularity,
            'bbox': (x, y, w_box, h_box)
        })

        # Draw on mask
        cv2.drawContours(mask, [contour], -1, 255, -1)

    if debug:
        debug_img = frame.copy()

        # Draw syringe outline in blue
        syringe_contours, _ = cv2.findContours(syringe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug_img, syringe_contours, -1, (255, 0, 0), 2)

        # Draw bubbles in green
        for bubble in bubbles:
            x, y, w_box, h_box = bubble['bbox']
            cv2.rectangle(debug_img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
            # Add circularity label
            cv2.putText(debug_img, f"{bubble['circularity']:.2f}",
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return mask, len(bubbles), debug_img, syringe_mask, bubble_candidates

    return mask, len(bubbles)

def process_bubbles3_video():
    """
    Process AIH_Bubbles3.mp4 and generate high-quality training masks.
    """
    print("=" * 80)
    print(" Processing AIH_Bubbles3.mp4 (Black Background)")
    print(" Advanced CV Pipeline with High-Contrast Detection")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parents[2]
    video_path = base_dir / "videos" / "AIH_Bubbles3.mp4"

    if not video_path.exists():
        print(f"\n✗ Video not found: {video_path}")
        return 0

    # Output directories
    output_dir = base_dir / "data" / "cnn_bubbles3"
    img_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    debug_dir = output_dir / "debug"
    analysis_dir = output_dir / "analysis"

    for d in [img_dir, mask_dir, debug_dir, analysis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nVideo: {video_path}")
    print(f"Output: {output_dir}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nVideo Info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Duration: {total_frames/fps:.1f}s")

    # Sample every 15 frames (~0.5 second intervals for 30fps)
    frame_interval = 15
    sample_frames = list(range(0, total_frames, frame_interval))

    print(f"\nSampling strategy:")
    print(f"  Interval: {frame_interval} frames (~{frame_interval/fps:.2f}s)")
    print(f"  Total samples: {len(sample_frames)} frames")

    total_bubbles = 0
    samples_created = 0
    bubble_counts = []

    print("\nProcessing frames...")
    for frame_idx in tqdm(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Detect bubbles
        mask, bubble_count, debug_img, syringe_mask, bubble_candidates = \
            detect_bubbles_black_bg(frame, debug=True)

        total_bubbles += bubble_count
        bubble_counts.append(bubble_count)

        # Save debug visualization
        debug_path = debug_dir / f"frame_{frame_idx:04d}_debug.png"
        cv2.imwrite(str(debug_path), debug_img)

        # Save analysis images
        cv2.imwrite(str(analysis_dir / f"frame_{frame_idx:04d}_syringe_mask.png"), syringe_mask)
        cv2.imwrite(str(analysis_dir / f"frame_{frame_idx:04d}_bubble_candidates.png"), bubble_candidates)

        # Create 256x256 crops
        h, w = frame.shape[:2]
        crop_size = 256
        stride = 128

        for y in range(0, h - crop_size + 1, stride):
            for x in range(0, w - crop_size + 1, stride):
                img_crop = frame[y:y+crop_size, x:x+crop_size]
                mask_crop = mask[y:y+crop_size, x:x+crop_size]

                # Save all crops (including negative samples)
                sample_id = f"bubbles3_frame_{frame_idx:04d}_y{y}_x{x}"
                cv2.imwrite(str(img_dir / f"{sample_id}.png"), img_crop)
                cv2.imwrite(str(mask_dir / f"{sample_id}.png"), mask_crop)
                samples_created += 1

    cap.release()

    # Statistics
    print("\n" + "=" * 80)
    print(" Processing Complete")
    print("=" * 80)
    print(f"Frames processed: {len(sample_frames)}")
    print(f"Total bubbles detected: {total_bubbles}")
    print(f"Average bubbles per frame: {np.mean(bubble_counts):.1f}")
    print(f"Min/Max bubbles: {min(bubble_counts)}/{max(bubble_counts)}")
    print(f"Training samples created: {samples_created}")
    print(f"\nSaved to:")
    print(f"  Images: {img_dir}/")
    print(f"  Masks:  {mask_dir}/")
    print(f"  Debug:  {debug_dir}/ (check quality)")
    print(f"  Analysis: {analysis_dir}/ (intermediate steps)")
    print("=" * 80)

    return samples_created

if __name__ == "__main__":
    process_bubbles3_video()
