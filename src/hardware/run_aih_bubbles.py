from pathlib import Path
import cv2
import time
import sys
import numpy as np
import argparse

# Add model path
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

# Paths
VIDEO_PATH = Path(__file__).resolve().parents[2] / "videos" / "AIH_Bubbles.mp4"
CKPT_PATH = Path(__file__).resolve().parents[2] / "data" / "cnn" / "small_unet_real_trained.pt"

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="M1 Bubble Detection")
    parser.add_argument("--backend", choices=["mps", "coreml"], default="mps",
                       help="Backend to use: mps (PyTorch MPS) or coreml (CoreML Neural Engine)")
    parser.add_argument("--video", type=str, default=str(VIDEO_PATH),
                       help="Path to input video")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to output video (default: input_processed.mp4)")
    args = parser.parse_args()

    # Set output path
    if args.output is None:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_processed_{args.backend}.mp4")

    # Load model based on backend
    if args.backend == "coreml":
        from bubble_coreml_model import BubbleCoreMLModel
        mlpackage_path = CKPT_PATH.parent / "small_unet_real_fp16.mlpackage"
        phase_name = "Phase 2 (CoreML Neural Engine - Real-Data Retrained)"

        print("=" * 70)
        print(f" M1 Bubble Detection - {phase_name}")
        print("=" * 70)
        print(f"\nLoading CoreML model: {mlpackage_path}")
        model = BubbleCoreMLModel(mlpackage_path, min_diam_px=5)
    else:
        from bubble_cnn_model import BubbleCNNModel
        phase_name = "Phase 1 (PyTorch MPS)"

        print("=" * 70)
        print(f" M1 Bubble Detection - {phase_name}")
        print("=" * 70)
        print(f"\nLoading PyTorch model: {CKPT_PATH}")
        model = BubbleCNNModel(CKPT_PATH, min_diam_px=5)

    print(f"✓ Device: {model.device}")

    # Open video
    print(f"\nOpening video: {args.video}")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {args.video}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"✓ Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
    print(f"✓ Output: {args.output}")

    # Processing
    print(f"\nProcessing frames...")
    print("-" * 70)
    frame_times = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Time the inference
            start = time.perf_counter()
            mask, bubbles = model.predict(frame)
            inference_time = (time.perf_counter() - start) * 1000  # ms
            frame_times.append(inference_time)

            # Visualize (same as run_synth.py pattern)
            vis = frame.copy()

            # Draw green bounding boxes
            for b in bubbles:
                x, y, w, h = b["x"], b["y"], b["w"], b["h"]
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add performance overlay
            avg_fps = 1000 / (sum(frame_times[-30:]) / len(frame_times[-30:])) if frame_times else 0

            # Black background for text
            overlay = vis.copy()
            cv2.rectangle(overlay, (5, 5), (500, 135), (0, 0, 0), -1)
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

            cv2.putText(vis, f"Frame: {frame_idx+1}/{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, f"Bubbles: {len(bubbles)}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, f"FPS: {avg_fps:.1f}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, f"Device: {model.device}",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write frame
            out.write(vis)

            # Display (optional - can comment out for headless)
            cv2.imshow("Bubble Detection (M1)", vis)
            cv2.imshow("Binary Mask", mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⚠ Stopped by user (pressed 'q')")
                break

            frame_idx += 1

            # Progress every 30 frames
            if frame_idx % 30 == 0:
                progress = 100 * frame_idx / total_frames
                print(f"Progress: {frame_idx}/{total_frames} ({progress:.1f}%) | Avg FPS: {avg_fps:.1f} | Inference: {np.mean(frame_times[-30:]):.2f}ms")

    except KeyboardInterrupt:
        print("\n⚠ Interrupted by user (Ctrl+C)")

    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Final stats
        if frame_times:
            print("-" * 70)
            print("\n" + "=" * 70)
            print(" Performance Report")
            print("=" * 70)
            print(f"Total frames processed: {len(frame_times)}/{total_frames}")
            print(f"Average inference time: {np.mean(frame_times):.2f} ms")
            print(f"Median inference time:  {np.median(frame_times):.2f} ms")
            print(f"Min/Max inference:      {np.min(frame_times):.2f} / {np.max(frame_times):.2f} ms")
            print(f"Average FPS:            {1000/np.mean(frame_times):.1f}")
            print(f"Device used:            {model.device}")
            print(f"\nOutput saved to: {args.output}")
            print("=" * 70)

            # Evaluation
            avg_fps_final = 1000 / np.mean(frame_times)
            print(f"\nPerformance Evaluation ({args.backend.upper()}):")
            if avg_fps_final >= 100:
                print(f"✓ EXCELLENT: {avg_fps_final:.1f} FPS - Matches/exceeds Hailo-8L (60-120 FPS)!")
            elif avg_fps_final >= 60:
                print(f"✓ VERY GOOD: {avg_fps_final:.1f} FPS - Target achieved! (≥60 FPS)")
                if args.backend == "mps":
                    print("  Try --backend coreml for even better performance (100-120 FPS)")
            elif avg_fps_final >= 40:
                print(f"✓ GOOD: {avg_fps_final:.1f} FPS - Better than real-time")
                if args.backend == "mps":
                    print("  Try --backend coreml to reach 100+ FPS for Hailo-8L match.")
            else:
                print(f"⚠ MODERATE: {avg_fps_final:.1f} FPS")
            print("=" * 70)

if __name__ == "__main__":
    main()
