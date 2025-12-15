#!/usr/bin/env python3
"""
Export clean frames from both videos for manual bubble labeling.
User will circle bubbles, then we'll convert annotations to masks.
"""
import cv2
import numpy as np
from pathlib import Path

def export_frames_for_labeling():
    print("=" * 80)
    print(" Exporting Frames for Manual Bubble Labeling")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parents[2]
    output_dir = base_dir / "manual_labeling_frames"
    output_dir.mkdir(exist_ok=True)

    # Videos to process
    videos = {
        "AIH_Bubbles": base_dir / "videos" / "AIH_Bubbles.mp4",
        "AIH_Bubbles2": base_dir / "videos" / "AIH_Bubbles2.mp4"
    }

    # Frame indices to export (5 diverse frames from each video)
    frame_indices = [30, 60, 90, 120, 150]

    for video_name, video_path in videos.items():
        print(f"\n{video_name}:")
        print(f"  Video: {video_path}")

        if not video_path.exists():
            print(f"  ✗ Video not found, skipping")
            continue

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for idx in frame_indices:
            if idx >= total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                continue

            # Save clean frame
            output_path = output_dir / f"{video_name}_frame_{idx:03d}.png"
            cv2.imwrite(str(output_path), frame)
            print(f"  ✓ Saved frame {idx:03d} → {output_path.name}")

        cap.release()

    print("\n" + "=" * 80)
    print(" Export Complete")
    print("=" * 80)
    print(f"\nFrames saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Open each image in an image editor")
    print("  2. Draw RED circles around EVERY bubble you see")
    print("  3. Save the annotated images with '_annotated' suffix:")
    print(f"     Example: AIH_Bubbles_frame_030_annotated.png")
    print("  4. Run convert_annotations_to_masks.py to generate training masks")
    print("=" * 80)

if __name__ == "__main__":
    export_frames_for_labeling()
