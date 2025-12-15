#!/usr/bin/env python3
"""
STRICT bubble detection with aggressive filtering
- Spatial exclusion zones (edges + top)
- Minimum bubble size
- High CNN confidence required
- Motion tracking for moving objects only
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import sys

# Add model directory to path
sys.path.append(str(Path(__file__).parent.parent / "model"))
from train_manual_cnn_balanced import SmallUNet

class StrictBubbleTracker:
    """Aggressive tracker - only large, moving, high-confidence detections"""

    def __init__(self,
                 max_distance=80,
                 min_movement=30,
                 history_frames=5,
                 exclude_top_pct=0.15,
                 exclude_side_pct=0.08,
                 min_bubble_area=500):
        """
        Args:
            max_distance: Max distance to match bubbles between frames
            min_movement: Minimum movement (pixels) - INCREASED
            history_frames: Number of frames to track
            exclude_top_pct: Exclude top % of frame
            exclude_side_pct: Exclude left/right % (syringe edges)
            min_bubble_area: Minimum bubble area in pixels
        """
        self.max_distance = max_distance
        self.min_movement = min_movement
        self.history_frames = history_frames
        self.exclude_top_pct = exclude_top_pct
        self.exclude_side_pct = exclude_side_pct
        self.min_bubble_area = min_bubble_area

        self.tracked_bubbles = {}
        self.next_id = 0
        self.frame_width = None
        self.frame_height = None

    def set_frame_size(self, width, height):
        """Set frame dimensions for exclusion zones"""
        self.frame_width = width
        self.frame_height = height
        self.exclude_top_y = int(height * self.exclude_top_pct)
        self.exclude_left_x = int(width * self.exclude_side_pct)
        self.exclude_right_x = int(width * (1 - self.exclude_side_pct))

    def is_in_valid_zone(self, centroid, bbox):
        """Check if bubble is in valid detection zone"""
        cx, cy = centroid
        x, y, w, h = bbox
        area = w * h

        # Check size
        if area < self.min_bubble_area:
            return False

        # Check top exclusion
        if cy < self.exclude_top_y:
            return False

        # Check side exclusions (syringe edges)
        if cx < self.exclude_left_x or cx > self.exclude_right_x:
            return False

        return True

    def update(self, bubble_centroids, frame_idx):
        """
        Update tracker with new bubbles - STRICT filtering

        Returns:
            moving_bubbles: List of (centroid, bbox) for VALID moving bubbles
        """
        if len(bubble_centroids) == 0:
            return []

        # Filter by spatial zones and size FIRST
        valid_bubbles = []
        for centroid, bbox in bubble_centroids:
            if self.is_in_valid_zone(centroid, bbox):
                valid_bubbles.append((centroid, bbox))

        if len(valid_bubbles) == 0:
            return []

        # Match with tracked bubbles
        matched_ids = set()
        unmatched_bubbles = []

        for centroid, bbox in valid_bubbles:
            best_id = None
            best_dist = self.max_distance

            for track_id, history in self.tracked_bubbles.items():
                if track_id in matched_ids:
                    continue

                last_centroid, last_frame = history[-1]

                # Recent frames only
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
                unmatched_bubbles.append((centroid, bbox))

        # Add new tracks
        for centroid, bbox in unmatched_bubbles:
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

        # Determine MOVING bubbles with STRICT criteria
        moving_bubbles = []

        for centroid, bbox in valid_bubbles:
            is_moving = False
            has_enough_history = False

            for track_id, history in self.tracked_bubbles.items():
                if len(history) < 3:  # Need at least 3 frames of history
                    continue

                # Check if this is the same bubble
                for tracked_centroid, _ in history[-2:]:  # Check last 2 positions
                    dist = np.sqrt((centroid[0] - tracked_centroid[0])**2 +
                                  (centroid[1] - tracked_centroid[1])**2)

                    if dist < 30:  # Same bubble
                        has_enough_history = True
                        # Check TOTAL movement across history
                        first_pos = history[0][0]
                        last_pos = history[-1][0]
                        total_movement = np.sqrt(
                            (last_pos[0] - first_pos[0])**2 +
                            (last_pos[1] - first_pos[1])**2
                        )

                        if total_movement > self.min_movement:
                            is_moving = True
                        break

                if has_enough_history:
                    break

            # Only count if has movement history
            if is_moving:
                moving_bubbles.append((centroid, bbox))

        return moving_bubbles

def group_contours_strict(binary_mask, dilation_kernel_size=20, min_confidence=0.6):
    """Group contours with STRICT criteria"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 500:  # Larger minimum
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                bbox = cv2.boundingRect(contour)
                bubbles.append(((cx, cy), bbox))

    return bubbles

def create_detection_video_strict():
    """Process with STRICT filtering"""

    base_dir = Path(__file__).parent.parent.parent
    video_path = base_dir / "videos" / "AIH_Bubbles3.mp4"
    model_path = base_dir / "data" / "cnn" / "small_unet_combined_trained.pt"
    output_dir = base_dir / "data" / "validation_bubbles3"
    output_video = output_dir / "AIH_Bubbles3_STRICT.mp4"

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {model_path}")
    model = SmallUNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\nSTRICT FILTERING ENABLED:")
    print(f"  - Minimum bubble size: 500px")
    print(f"  - Minimum movement: 30px")
    print(f"  - Top exclusion: 15%")
    print(f"  - Side exclusion: 8% (edges)")
    print(f"  - Requires 3+ frames of history")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo: {video_path.name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")

    # Initialize tracker
    tracker = StrictBubbleTracker(
        max_distance=80,
        min_movement=30,
        history_frames=5,
        exclude_top_pct=0.15,
        exclude_side_pct=0.08,
        min_bubble_area=500
    )
    tracker.set_frame_size(width, height)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    tile_size = 256
    stride = 128
    frame_idx = 0
    total_bubbles = 0

    print("\nProcessing with STRICT filtering...")

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

        # Higher threshold for binary mask
        binary_mask = (full_mask > 0.6).astype(np.uint8) * 255

        # Group with strict criteria
        all_bubbles = group_contours_strict(binary_mask, dilation_kernel_size=20)

        # Filter with motion tracking
        moving_bubbles = tracker.update(all_bubbles, frame_idx)
        num_bubbles = len(moving_bubbles)
        total_bubbles += num_bubbles

        # Visualization
        vis = frame.copy()

        # Draw exclusion zones
        cv2.line(vis, (0, tracker.exclude_top_y), (w, tracker.exclude_top_y), (255, 0, 0), 2)
        cv2.line(vis, (tracker.exclude_left_x, 0), (tracker.exclude_left_x, h), (255, 0, 0), 2)
        cv2.line(vis, (tracker.exclude_right_x, 0), (tracker.exclude_right_x, h), (255, 0, 0), 2)

        # Draw mask overlay (very faint)
        mask_overlay = np.zeros_like(vis)
        mask_overlay[binary_mask > 0] = [0, 255, 0]
        vis = cv2.addWeighted(vis, 0.9, mask_overlay, 0.1, 0)

        # Draw ONLY strictly validated moving bubbles
        for centroid, bbox in moving_bubbles:
            x, y, w_box, h_box = bbox
            cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), (0, 255, 255), 4)
            cv2.circle(vis, centroid, 8, (0, 0, 255), -1)

        # Text
        text = f"Frame {frame_idx} | STRICT Bubbles: {num_bubbles}"
        cv2.putText(vis, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   1.2, (0, 255, 255), 3, cv2.LINE_AA)

        out.write(vis)

        if frame_idx % 20 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"  Frame {frame_idx}/{total_frames} ({progress:.1f}%) - {num_bubbles} bubbles")

        frame_idx += 1

    cap.release()
    out.release()

    avg_bubbles = total_bubbles / frame_idx if frame_idx > 0 else 0

    print(f"\n{'='*60}")
    print(f"STRICT filtering complete!")
    print(f"  Processed frames: {frame_idx}")
    print(f"  Total bubbles: {total_bubbles}")
    print(f"  Average: {avg_bubbles:.2f} bubbles/frame")
    print(f"  Output: {output_video}")
    print(f"{'='*60}\n")

    return output_video

if __name__ == "__main__":
    output = create_detection_video_strict()
    if output:
        import subprocess
        subprocess.run(["open", str(output.parent)])
