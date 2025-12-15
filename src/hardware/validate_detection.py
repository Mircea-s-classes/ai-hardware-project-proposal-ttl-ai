#!/usr/bin/env python3
"""
COMPREHENSIVE VALIDATION: Test perfect model on all three bubble videos
- AIH_Bubbles.mp4
- AIH_Bubbles2.mp4
- AIH_Bubbles3.mp4

Uses FINAL detection pipeline:
- Motion tracking (20px minimum movement)
- Static filtering (variance < 10)
- Size filtering (3000px minimum)
- Top exclusion (15%)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import sys
from collections import defaultdict

# Add model directory to path
sys.path.append(str(Path(__file__).parent.parent / "model"))
from train_manual_cnn_balanced import SmallUNet

class BubbleTracker:
    """Track bubbles across frames - FINAL PERFECT VERSION"""

    def __init__(self, max_distance=100, min_movement=20, history_frames=5,
                 exclude_top_pct=0.15, min_bubble_area=3000):
        """
        Args:
            max_distance: Max distance to match bubbles between frames
            min_movement: Minimum movement (pixels) - 20px for real bubbles
            history_frames: Number of frames to track history
            exclude_top_pct: Exclude top % of frame
            min_bubble_area: 3000px minimum to filter small false positives
        """
        self.max_distance = max_distance
        self.min_movement = min_movement
        self.history_frames = history_frames
        self.exclude_top_pct = exclude_top_pct
        self.min_bubble_area = min_bubble_area

        self.tracked_bubbles = {}
        self.next_id = 0
        self.frame_height = None

    def set_frame_height(self, height):
        """Set frame height for top exclusion"""
        self.frame_height = height
        self.exclude_top_y = int(height * self.exclude_top_pct)

    def reset(self):
        """Reset tracker for new video"""
        self.tracked_bubbles = {}
        self.next_id = 0

    def update(self, bubble_centroids, frame_idx):
        """
        Update tracker with new bubbles - FINAL VERSION with all filters

        Returns:
            moving_bubbles: List of (centroid, bbox) for bubbles that are MOVING
        """
        if len(bubble_centroids) == 0:
            return []

        # Filter by size and top exclusion FIRST
        valid_bubbles = []
        for centroid, bbox in bubble_centroids:
            x, y, w, h = bbox
            area = w * h

            # Size filter - 3000px minimum
            if area < self.min_bubble_area:
                continue

            # Skip top exclusion zone
            if centroid[1] < self.exclude_top_y:
                continue

            valid_bubbles.append((centroid, bbox))

        if len(valid_bubbles) == 0:
            return []

        # Match current bubbles with tracked ones
        matched_ids = set()
        unmatched_centroids = []

        for centroid, bbox in valid_bubbles:
            best_id = None
            best_dist = self.max_distance

            for track_id, history in self.tracked_bubbles.items():
                if track_id in matched_ids:
                    continue

                last_centroid, last_frame = history[-1]

                if frame_idx - last_frame > 3:
                    continue

                dist = np.sqrt((centroid[0] - last_centroid[0])**2 +
                              (centroid[1] - last_centroid[1])**2)

                if dist < best_dist:
                    best_dist = dist
                    best_id = track_id

            if best_id is not None:
                self.tracked_bubbles[best_id].append((centroid, frame_idx))
                matched_ids.add(best_id)
            else:
                unmatched_centroids.append((centroid, bbox))

        # Add new tracks
        for centroid, bbox in unmatched_centroids:
            self.tracked_bubbles[self.next_id] = [(centroid, frame_idx)]
            self.next_id += 1

        # Prune old tracks
        to_remove = []
        for track_id, history in self.tracked_bubbles.items():
            self.tracked_bubbles[track_id] = [
                (c, f) for c, f in history
                if frame_idx - f < self.history_frames
            ]

            if len(self.tracked_bubbles[track_id]) == 0:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracked_bubbles[track_id]

        # Determine which bubbles are MOVING with static detection
        moving_bubbles = []

        for centroid, bbox in valid_bubbles:
            is_moving = False
            is_static = False

            for track_id, history in self.tracked_bubbles.items():
                # Check if this is the same bubble
                for tracked_centroid, _ in history[-2:]:
                    dist = np.sqrt((centroid[0] - tracked_centroid[0])**2 +
                                  (centroid[1] - tracked_centroid[1])**2)

                    if dist < 20:  # Same bubble
                        # Check if object is STATIC
                        if len(history) >= 4:
                            positions = np.array([c for c, _ in history])
                            variance = np.var(positions, axis=0)
                            total_variance = variance[0] + variance[1]

                            # If position variance is very low, it's static
                            if total_variance < 10:
                                is_static = True
                                break

                        # Check total movement
                        if len(history) >= 2:
                            first_pos = history[0][0]
                            last_pos = history[-1][0]
                            total_movement = np.sqrt(
                                (last_pos[0] - first_pos[0])**2 +
                                (last_pos[1] - first_pos[1])**2
                            )

                            if total_movement > self.min_movement:
                                is_moving = True
                        else:
                            # New detection, assume moving for now
                            is_moving = True
                        break

                if is_static or is_moving:
                    break

            # Only count if moving AND not static
            if is_moving and not is_static:
                moving_bubbles.append((centroid, bbox))

        return moving_bubbles


def group_contours_into_bubbles(binary_mask, dilation_kernel_size=15):
    """Group nearby contours into bubble clusters"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 100:  # Initial minimum
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                bbox = cv2.boundingRect(contour)
                bubbles.append(((cx, cy), bbox))

    return bubbles


def process_video(video_path, model, device, tracker, output_video_path=None):
    """Process a single video and return metrics"""

    print(f"\n{'='*70}")
    print(f"PROCESSING: {video_path.name}")
    print(f"{'='*70}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: Could not open video")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")

    # Reset tracker for new video
    tracker.reset()
    tracker.set_frame_height(height)

    # Setup video writer if output path provided
    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    tile_size = 256
    stride = 128
    frame_idx = 0
    total_bubbles = 0
    bubbles_per_frame = []

    print("\nProcessing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        # Threshold
        binary_mask = (full_mask > 0.5).astype(np.uint8) * 255

        # Group bubbles
        all_bubbles = group_contours_into_bubbles(binary_mask, dilation_kernel_size=15)

        # Filter with motion tracking and all filters
        moving_bubbles = tracker.update(all_bubbles, frame_idx)
        num_bubbles = len(moving_bubbles)
        total_bubbles += num_bubbles
        bubbles_per_frame.append(num_bubbles)

        # Generate visualization if output requested
        if out:
            vis = frame.copy()

            # Draw exclusion zone
            cv2.line(vis, (0, tracker.exclude_top_y), (w, tracker.exclude_top_y), (255, 0, 0), 2)
            cv2.putText(vis, "EXCLUSION ZONE", (10, tracker.exclude_top_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Draw mask overlay (green, faint)
            mask_overlay = np.zeros_like(vis)
            mask_overlay[binary_mask > 0] = [0, 255, 0]
            vis = cv2.addWeighted(vis, 0.85, mask_overlay, 0.15, 0)

            # Draw moving bubbles
            for centroid, bbox in moving_bubbles:
                x, y, w_box, h_box = bbox
                cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), (0, 255, 255), 3)
                cv2.circle(vis, centroid, 5, (0, 0, 255), -1)

            # Text overlay
            text = f"Frame {frame_idx} | Bubbles: {num_bubbles}"
            cv2.putText(vis, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (0, 255, 255), 3, cv2.LINE_AA)

            out.write(vis)

        if frame_idx % 20 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"  Frame {frame_idx}/{total_frames} ({progress:.1f}%) - {num_bubbles} bubbles")

        frame_idx += 1

    cap.release()
    if out:
        out.release()

    # Calculate metrics
    avg_bubbles = total_bubbles / frame_idx if frame_idx > 0 else 0
    max_bubbles = max(bubbles_per_frame) if bubbles_per_frame else 0
    min_bubbles = min(bubbles_per_frame) if bubbles_per_frame else 0
    std_bubbles = np.std(bubbles_per_frame) if bubbles_per_frame else 0

    metrics = {
        'video_name': video_path.name,
        'total_frames': frame_idx,
        'total_bubbles': total_bubbles,
        'avg_bubbles': avg_bubbles,
        'max_bubbles': max_bubbles,
        'min_bubbles': min_bubbles,
        'std_bubbles': std_bubbles,
        'fps': fps,
        'resolution': f"{width}x{height}"
    }

    print(f"\n{'='*70}")
    print(f"RESULTS: {video_path.name}")
    print(f"{'='*70}")
    print(f"  Total frames: {frame_idx}")
    print(f"  Total bubbles detected: {total_bubbles}")
    print(f"  Average bubbles/frame: {avg_bubbles:.2f}")
    print(f"  Max bubbles/frame: {max_bubbles}")
    print(f"  Min bubbles/frame: {min_bubbles}")
    print(f"  Std deviation: {std_bubbles:.2f}")
    if output_video_path:
        print(f"  Output video: {output_video_path}")
    print(f"{'='*70}\n")

    return metrics


def validate_all_videos():
    """Validate perfect model on all three bubble videos"""

    base_dir = Path(__file__).parent.parent.parent
    videos_dir = base_dir / "videos"
    model_path = base_dir / "data" / "cnn" / "small_unet_combined_trained.pt"
    output_dir = base_dir / "data" / "validation_all"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"{'='*70}")
    print(f"COMPREHENSIVE VALIDATION - PERFECT MODEL")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Model: {model_path.name}")
    print(f"\nDETECTION PIPELINE:")
    print(f"  - Motion tracking: 20px minimum movement")
    print(f"  - Static filtering: variance < 10")
    print(f"  - Size filtering: 3000px minimum area")
    print(f"  - Top exclusion: 15%")
    print(f"  - Dilation kernel: 15px")

    # Load model
    print(f"\nLoading perfect model...")
    model = SmallUNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✓ Model loaded (Dice: {checkpoint.get('best_val_dice', 'N/A')})")

    # Initialize tracker
    tracker = BubbleTracker(
        max_distance=100,
        min_movement=20,
        history_frames=5,
        exclude_top_pct=0.15,
        min_bubble_area=3000
    )

    # Process all three videos
    video_files = [
        "AIH_Bubbles.mp4",
        "AIH_Bubbles2.mp4",
        "AIH_Bubbles3.mp4"
    ]

    all_metrics = []

    for video_name in video_files:
        video_path = videos_dir / video_name

        if not video_path.exists():
            print(f"⚠ Warning: {video_name} not found, skipping...")
            continue

        # Create output video name
        output_video_name = video_name.replace(".mp4", "_VALIDATED.mp4")
        output_video_path = output_dir / output_video_name

        # Process video
        metrics = process_video(
            video_path,
            model,
            device,
            tracker,
            output_video_path
        )

        if metrics:
            all_metrics.append(metrics)

    # Summary report
    print(f"\n{'='*70}")
    print(f"VALIDATION SUMMARY - ALL VIDEOS")
    print(f"{'='*70}\n")

    for metrics in all_metrics:
        print(f"{metrics['video_name']:25} | "
              f"{metrics['total_frames']:4d} frames | "
              f"{metrics['total_bubbles']:5d} bubbles | "
              f"{metrics['avg_bubbles']:5.2f} avg/frame | "
              f"{metrics['max_bubbles']:3d} max | "
              f"σ={metrics['std_bubbles']:.2f}")

    print(f"\n{'='*70}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*70}")

    total_frames_all = sum(m['total_frames'] for m in all_metrics)
    total_bubbles_all = sum(m['total_bubbles'] for m in all_metrics)
    avg_all = total_bubbles_all / total_frames_all if total_frames_all > 0 else 0

    print(f"  Total frames processed: {total_frames_all}")
    print(f"  Total bubbles detected: {total_bubbles_all}")
    print(f"  Overall average: {avg_all:.2f} bubbles/frame")
    print(f"  Videos validated: {len(all_metrics)}/3")
    print(f"\nOutput directory: {output_dir}")
    print(f"{'='*70}\n")

    return all_metrics


if __name__ == "__main__":
    try:
        metrics = validate_all_videos()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
