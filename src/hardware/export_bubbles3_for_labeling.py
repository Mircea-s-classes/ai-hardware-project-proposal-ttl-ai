#!/usr/bin/env python3
"""
Export 15 frames from AIH_Bubbles3.mp4 for manual annotation.
User will label complete bubble regions, not just bright reflections.
"""
import cv2
import numpy as np
from pathlib import Path

def export_frames():
    print("=" * 80)
    print(" Exporting 15 Frames from AIH_Bubbles3.mp4 for Manual Labeling")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parents[2]
    video_path = base_dir / "videos" / "AIH_Bubbles3.mp4"
    output_dir = base_dir / "manual_labeling_bubbles3"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"✗ Video not found: {video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nVideo: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps:.1f}")
    print(f"Duration: {total_frames/fps:.1f}s")

    # Sample 15 frames evenly spaced throughout video
    # Frame indices: 0, 12, 24, 36, ..., 168
    frame_indices = [i * 12 for i in range(15)]

    print(f"\nExporting {len(frame_indices)} frames...")
    print(f"Frame indices: {frame_indices}")
    print(f"Output directory: {output_dir}")

    exported = 0
    for frame_idx in frame_indices:
        if frame_idx >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        output_path = output_dir / f"Bubbles3_frame_{frame_idx:03d}.png"
        cv2.imwrite(str(output_path), frame)
        exported += 1

    cap.release()

    print(f"\n✓ Exported {exported} frames")
    print(f"\nInstructions:")
    print(f"  1. Open: {output_dir}")
    print(f"  2. Draw BLACK boxes around ENTIRE bubbles")
    print(f"     (not just the bright reflections)")
    print(f"  3. Include the bubble body, not just the bright spot")
    print(f"  4. Save files in the same directory")
    print("=" * 80)

if __name__ == "__main__":
    export_frames()
