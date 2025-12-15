import argparse
from pathlib import Path
import glob
import numpy as np
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="folder containing extracted frames (png/jpg)")
    ap.add_argument("--out_mask", required=True, help="output path for syringe ROI mask png")
    ap.add_argument("--n", type=int, default=60, help="number of frames to use for median")
    ap.add_argument("--invert", action="store_true", help="invert Otsu result if needed")
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        raise SystemExit(f"frames_dir does not exist: {frames_dir}")

    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths += sorted(glob.glob(str(frames_dir / ext)))

    print(f"[roi] frames_dir: {frames_dir}")
    print(f"[roi] found {len(paths)} frames")
    if len(paths) == 0:
        raise SystemExit("No frames found. Check folder name and extensions.")

    paths = paths[: min(args.n, len(paths))]
    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            imgs.append(im)

    print(f"[roi] loaded {len(imgs)} frames for median")
    if len(imgs) < 10:
        raise SystemExit("Too few readable frames to build ROI mask.")

    med = np.median(np.stack(imgs, axis=0), axis=0).astype(np.uint8)

    # Otsu split: black bg -> syringe interior usually becomes white
    _, roi = cv2.threshold(med, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if args.invert:
        roi = cv2.bitwise_not(roi)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k, iterations=2)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, k, iterations=1)

    out_mask = Path(args.out_mask)
    out_mask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_mask), roi)

    # Debug saves
    cv2.imwrite(str(out_mask.with_name(out_mask.stem + "_median.png")), med)
    cv2.imwrite(str(out_mask.with_name(out_mask.stem + "_debug.png")), roi)

    print(f"[roi] wrote: {out_mask}")

if __name__ == "__main__":
    main()
