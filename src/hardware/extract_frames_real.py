from pathlib import Path
import argparse
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/raw/real_water")
    ap.add_argument("--out_dir", default="data/processed/real_water/frames")
    ap.add_argument("--every_n", type=int, default=5)
    ap.add_argument("--resize", type=int, default=512, help="square resize; 0 keeps original")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    vids = sorted(list(in_dir.glob("*.mp4")) + list(in_dir.glob("*.mov")) + list(in_dir.glob("*.mkv")))
    if not vids:
        raise SystemExit(f"No videos found in {in_dir}")

    for vp in vids:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"Skip (can't open): {vp}")
            continue

        clip_out = out_root / vp.stem
        clip_out.mkdir(parents=True, exist_ok=True)

        i = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % args.every_n != 0:
                i += 1
                continue

            # grayscale for easier labeling + consistent training
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if args.resize and args.resize > 0:
                gray = cv2.resize(gray, (args.resize, args.resize), interpolation=cv2.INTER_AREA)

            outp = clip_out / f"frame_{saved:06d}.png"
            cv2.imwrite(str(outp), gray)
            saved += 1
            i += 1

        cap.release()
        print(f"[extract] {vp.name}: saved {saved} frames -> {clip_out}")

if __name__ == "__main__":
    main()
