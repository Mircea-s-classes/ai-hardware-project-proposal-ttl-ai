#!/usr/bin/env python3
"""
Create video visualization with MOTION TRACKING
Only counts MOVING objects as bubbles, filters out static syringe markings
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
    """Track bubbles across frames to filter static objects"""

    def __init__(self, max_distance=100, min_movement=20, history_frames=5, exclude_top_pct=0.15):
        """
        Args:
            max_distance: Max distance to match bubbles between frames
            min_movement: Minimum movement (pixels) to be considered a moving bubble
            history_frames: Number of frames to track history
            exclude_top_pct: Exclude top % of frame (bubbles already counted)
        """
        self.max_distance = max_distance
        self.min_movement = min_movement
        self.history_frames = history_frames
        self.exclude_top_pct = exclude_top_pct

        self.tracked_bubbles = {}  # id -> list of (centroid, frame_idx)
        self.next_id = 0
        self.frame_height = None

    def set_frame_height(self, height):
        """Set frame height for top exclusion"""
        self.frame_height = height
        self.exclude_top_y = int(height * self.exclude_top_pct)

    def update(self, bubble_centroids, frame_idx):
        """
        Update tracker with new bubbles

        Returns:
            moving_bubbles: List of (centroid, bbox) for bubbles that are MOVING
        """
        if len(bubble_centroids) == 0:
            return []

        # Match current bubbles with tracked ones
        matched_ids = set()
        unmatched_centroids = []

        for centroid, bbox in bubble_centroids:
            # Skip bubbles in top exclusion zone
            if centroid[1] < self.exclude_top_y:
                continue

            # Find closest tracked bubble
            best_id = None
            best_dist = self.max_distance

            for track_id, history in self.tracked_bubbles.items():
                if track_id in matched_ids:
                    continue

                last_centroid, last_frame = history[-1]

                # Only match if from recent frame
                if frame_idx - last_frame > 3:
                    continue

                dist = np.sqrt((centroid[0] - last_centroid[0])**2 +
                              (centroid[1] - last_centroid[1])**2)

                if dist < best_dist:
                    best_dist = dist
                    best_id = track_id

            if best_id is not None:
                # Update existing track
                self.tracked_bubbles[best_id].append((centroid, frame_idx))
                matched_ids.add(best_id)
            else:
                # New bubble
                unmatched_centroids.append((centroid, bbox))

        # Add new tracks for unmatched bubbles
        for centroid, bbox in unmatched_centroids:
            self.tracked_bubbles[self.next_id] = [(centroid, frame_idx)]
            self.next_id += 1

        # Prune old tracks and keep only recent history
        to_remove = []
        for track_id, history in self.tracked_bubbles.items():
            # Keep only recent history
            self.tracked_bubbles[track_id] = [
                (c, f) for c, f in history
                if frame_idx - f < self.history_frames
            ]

            # Remove empty tracks
            if len(self.tracked_bubbles[track_id]) == 0:
                to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracked_bubbles[track_id]

        # Determine which bubbles are MOVING
        moving_bubbles = []

        for centroid, bbox in bubble_centroids:
            # Skip top exclusion zone
            if centroid[1] < self.exclude_top_y:
                continue

            # Check if this bubble has moved
            is_moving = False

            for track_id, history in self.tracked_bubbles.items():
                if len(history) < 2:
                    # New bubble, assume moving for now
                    is_moving = True
                    break

                # Check if centroid matches any in this track
                for tracked_centroid, _ in history:
                    dist = np.sqrt((centroid[0] - tracked_centroid[0])**2 +
                                  (centroid[1] - tracked_centroid[1])**2)

                    if dist < 20:  # Same bubble
                        # Check total movement in track
                        first_pos = history[0][0]
                        last_pos = history[-1][0]
                        total_movement = np.sqrt(
                            (last_pos[0] - first_pos[0])**2 +
                            (last_pos[1] - first_pos[1])**2
                        )

                        if total_movement > self.min_movement:
                            is_moving = True
                        break

                if is_moving:
                    break

            if is_moving:
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
        if area >= 100:
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                bbox = cv2.boundingRect(contour)
                bubbles.append(((cx, cy), bbox))

    return bubbles

def create_detection_video_with_tracking():
    """Process AIH_Bubbles3.mp4 with motion tracking"""

    # Paths
    base_dir = Path(__file__).parent.parent.parent
    video_path = base_dir / "videos" / "AIH_Bubbles3.mp4"
    model_path = base_dir / "data" / "cnn" / "small_unet_combined_trained.pt"
    output_dir = base_dir / "data" / "validation_bubbles3"
    output_video = output_dir / "AIH_Bubbles3_TRACKED.mp4"

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
    print(f"\nMOTION TRACKING ENABLED:")
    print(f"  - Only MOVING objects counted as bubbles")
    print(f"  - Static syringe markings filtered out")
    print(f"  - Top 15% of frame excluded (already counted)")

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

    # Initialize tracker
    tracker = BubbleTracker(
        max_distance=100,
        min_movement=20,  # Minimum 20px movement to be considered a bubble
        history_frames=5,
        exclude_top_pct=0.15
    )
    tracker.set_frame_height(height)

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

    print("\nProcessing frames with MOTION TRACKING...")

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

        # Threshold to binary mask
        binary_mask = (full_mask > 0.5).astype(np.uint8) * 255

        # Group into bubble clusters
        all_bubbles = group_contours_into_bubbles(binary_mask, dilation_kernel_size=15)

        # Filter with motion tracking
        moving_bubbles = tracker.update(all_bubbles, frame_idx)
        num_bubbles = len(moving_bubbles)
        total_bubbles += num_bubbles

        # Create visualization
        vis = frame.copy()

        # Draw exclusion zone
        cv2.line(vis, (0, tracker.exclude_top_y), (w, tracker.exclude_top_y), (255, 0, 0), 2)
        cv2.putText(vis, "EXCLUSION ZONE", (10, tracker.exclude_top_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw original mask overlay (green, faint)
        mask_overlay = np.zeros_like(vis)
        mask_overlay[binary_mask > 0] = [0, 255, 0]
        vis = cv2.addWeighted(vis, 0.85, mask_overlay, 0.15, 0)

        # Draw ONLY moving bubbles (yellow, bright)
        for centroid, bbox in moving_bubbles:
            x, y, w_box, h_box = bbox
            cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), (0, 255, 255), 3)
            # Draw centroid
            cv2.circle(vis, centroid, 5, (0, 0, 255), -1)

        # Text overlay
        text = f"Frame {frame_idx} | MOVING Bubbles: {num_bubbles}"
        cv2.putText(vis, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 255, 255), 3, cv2.LINE_AA)

        # Write frame
        out.write(vis)

        # Progress update
        if frame_idx % 20 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"  Frame {frame_idx}/{total_frames} ({progress:.1f}%) - {num_bubbles} moving bubbles")

        frame_idx += 1

    # Cleanup
    cap.release()
    out.release()

    avg_bubbles = total_bubbles / frame_idx if frame_idx > 0 else 0

    print(f"\n{'='*60}")
    print(f"Video processing complete (MOTION TRACKING)!")
    print(f"  Processed frames: {frame_idx}")
    print(f"  Total MOVING bubbles: {total_bubbles}")
    print(f"  Average bubbles per frame: {avg_bubbles:.2f}")
    print(f"  Static objects filtered out: YES")
    print(f"  Top exclusion zone: {tracker.exclude_top_pct*100}%")
    print(f"  Output saved to: {output_video}")
    print(f"{'='*60}\n")

    return output_video

if __name__ == "__main__":
    output = create_detection_video_with_tracking()
    if output:
        print(f"Opening output directory...")
        import subprocess
        subprocess.run(["open", str(output.parent)])
