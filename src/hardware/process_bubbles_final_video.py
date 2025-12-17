#!/usr/bin/env python3
"""
Process AIH_Bubbles_Final.mp4 with PERFECT detection pipeline
Same settings that achieved 1.91 bubbles/frame on Bubbles3
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
    """FINAL PERFECT VERSION - Tracks bubbles across frames"""

    def __init__(self, max_distance=100, min_movement=20, history_frames=5,
                 exclude_top_pct=0.15, min_bubble_area=6000,
                 exclude_edge_pct=0.15):
        self.max_distance = max_distance
        self.min_movement = min_movement
        self.history_frames = history_frames
        self.exclude_top_pct = exclude_top_pct
        self.exclude_edge_pct = exclude_edge_pct
        self.min_bubble_area = min_bubble_area
        self.tracked_bubbles = {}
        self.next_id = 0
        self.frame_height = None
        self.frame_width = None

    def set_frame_dimensions(self, width, height):
        self.frame_width = width
        self.frame_height = height
        self.exclude_top_y = int(height * self.exclude_top_pct)
        self.exclude_left_x = int(width * self.exclude_edge_pct)
        self.exclude_right_x = int(width * (1 - self.exclude_edge_pct))

    def update(self, bubble_centroids, frame_idx):
        if len(bubble_centroids) == 0:
            return []

        # Filter by size, top exclusion, and edge exclusion
        valid_bubbles = []
        for centroid, bbox in bubble_centroids:
            x, y, w, h = bbox
            area = w * h

            # Size filter - INCREASED to 6000px
            if area < self.min_bubble_area:
                continue

            # Top exclusion zone
            if centroid[1] < self.exclude_top_y:
                continue

            # Edge exclusion zones (left/right 15%)
            if centroid[0] < self.exclude_left_x or centroid[0] > self.exclude_right_x:
                continue

            valid_bubbles.append((centroid, bbox))

        if len(valid_bubbles) == 0:
            return []

        # Match with tracked bubbles
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

        # Determine which bubbles are MOVING
        moving_bubbles = []

        for centroid, bbox in valid_bubbles:
            is_moving = False
            is_static = False

            for track_id, history in self.tracked_bubbles.items():
                for tracked_centroid, _ in history[-2:]:
                    dist = np.sqrt((centroid[0] - tracked_centroid[0])**2 +
                                  (centroid[1] - tracked_centroid[1])**2)

                    if dist < 20:
                        # Static detection - MORE STRICT (variance < 5)
                        if len(history) >= 4:
                            positions = np.array([c for c, _ in history])
                            variance = np.var(positions, axis=0)
                            total_variance = variance[0] + variance[1]
                            if total_variance < 5:  # Stricter threshold
                                is_static = True
                                break

                        # Movement check
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
                            is_moving = True
                        break

                if is_static or is_moving:
                    break

            if is_moving and not is_static:
                moving_bubbles.append((centroid, bbox))

        return moving_bubbles


def group_contours_into_bubbles(binary_mask, dilation_kernel_size=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 100:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                bbox = cv2.boundingRect(contour)
                bubbles.append(((cx, cy), bbox))

    return bubbles


def process_bubbles_final_video():
    """Process AIH_Bubbles_Final.mp4 with PERFECT pipeline"""

    base_dir = Path(__file__).parent.parent.parent
    video_path = base_dir / "videos" / "AIH_Bubbles_Final.mp4"
    model_path = base_dir / "data" / "cnn" / "small_unet_combined_trained.pt"
    output_dir = base_dir / "data" / "validation_bubbles_final"
    output_video = output_dir / "AIH_Bubbles_Final_PROCESSED.mp4"

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"{'='*70}")
    print(f"PROCESSING AIH_Bubbles_Final.mp4 WITH PERFECT PIPELINE")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Input: {video_path}")
    print(f"Output: {output_video}")

    # Load perfect model
    print(f"\nLoading perfect model...")
    model = SmallUNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✓ Model loaded (Dice: {checkpoint.get('best_val_dice', 'N/A')})")

    print(f"\nULTRA-STRICT PIPELINE SETTINGS:")
    print(f"  - Motion tracking: 20px minimum movement")
    print(f"  - Static filtering: variance < 5 (STRICTER)")
    print(f"  - Size filtering: 6000px minimum area (DOUBLED)")
    print(f"  - Top exclusion: 15%")
    print(f"  - Edge exclusion: 15% left/right (NEW)")
    print(f"  - Dilation kernel: 15px")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: Could not open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")

    # Initialize tracker with ULTRA-STRICT settings
    tracker = BubbleTracker(
        max_distance=100,
        min_movement=20,
        history_frames=5,
        exclude_top_pct=0.15,
        min_bubble_area=6000,  # DOUBLED from 3000
        exclude_edge_pct=0.15  # NEW: exclude edges
    )
    tracker.set_frame_dimensions(width, height)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    tile_size = 256
    stride = 128
    frame_idx = 0
    total_bubbles = 0

    print(f"\nProcessing frames...")

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

        # Filter with PERFECT pipeline
        moving_bubbles = tracker.update(all_bubbles, frame_idx)
        num_bubbles = len(moving_bubbles)
        total_bubbles += num_bubbles

        # Visualization
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
        text = f"Frame {frame_idx} | FINAL Bubbles: {num_bubbles}"
        cv2.putText(vis, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 255, 255), 3, cv2.LINE_AA)

        out.write(vis)

        if frame_idx % 50 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"  Frame {frame_idx}/{total_frames} ({progress:.1f}%) - {num_bubbles} bubbles")

        frame_idx += 1

    cap.release()
    out.release()

    avg_bubbles = total_bubbles / frame_idx if frame_idx > 0 else 0

    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"  Processed frames: {frame_idx}")
    print(f"  Total bubbles detected: {total_bubbles}")
    print(f"  Average: {avg_bubbles:.2f} bubbles/frame")
    print(f"  Output video: {output_video}")
    print(f"{'='*70}\n")

    return output_video, frame_idx, total_bubbles, avg_bubbles


if __name__ == "__main__":
    try:
        output_video, frames, total, avg = process_bubbles_final_video()
        print(f"✅ SUCCESS: Processed {frames} frames, detected {total} bubbles ({avg:.2f} avg)")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
