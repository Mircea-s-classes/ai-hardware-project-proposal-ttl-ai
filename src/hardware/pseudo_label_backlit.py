import argparse
from pathlib import Path
import glob
import numpy as np
import cv2

def fill_holes(bin_u8: np.ndarray) -> np.ndarray:
    """Fill holes inside a binary mask (0/255)."""
    h, w = bin_u8.shape
    ff = bin_u8.copy()
    tmp = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, tmp, (0, 0), 255)
    inv = cv2.bitwise_not(ff)
    return bin_u8 | inv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_masks_dir", required=True)
    ap.add_argument("--roi_mask", required=True)

    # Background + diff
    ap.add_argument("--n_bg", type=int, default=60, help="frames used for background median")
    ap.add_argument("--diff_thr", type=int, default=18, help="absdiff threshold")

    # Static artifact removal (tick marks/text)
    ap.add_argument("--static_ratio", type=float, default=0.60,
                    help="pixels that exceed diff_thr in > static_ratio of bg frames are treated as static")
    ap.add_argument("--static_dilate", type=int, default=1, help="dilate static mask to fully cover tick marks")

    # Morphology merge (reduce bubble splitting)
    ap.add_argument("--open_iters", type=int, default=1)
    ap.add_argument("--close_iters", type=int, default=3)
    ap.add_argument("--final_merge_close_iters", type=int, default=2)

    # Component filtering
    ap.add_argument("--min_area", type=int, default=30)
    ap.add_argument("--max_area", type=int, default=6000)
    ap.add_argument("--min_ellipticity", type=float, default=0.25, help="minor/major ratio (0..1)")
    ap.add_argument("--min_solidity", type=float, default=0.70, help="area/convexHullArea (0..1)")

    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_masks_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roi = cv2.imread(args.roi_mask, cv2.IMREAD_GRAYSCALE)
    if roi is None:
        raise SystemExit(f"Could not read roi_mask: {args.roi_mask}")
    roi = (roi > 127).astype(np.uint8) * 255

    # Accept png/jpg/jpeg
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths += sorted(glob.glob(str(frames_dir / ext)))
    if not paths:
        raise SystemExit(f"No frames found in {frames_dir}")

    # -------- Build background model --------
    bg_stack = []
    for p in paths[:min(args.n_bg, len(paths))]:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            bg_stack.append(im)
    if len(bg_stack) < 10:
        raise SystemExit("Too few frames to build background model.")
    bg = np.median(np.stack(bg_stack, 0), 0).astype(np.uint8)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # -------- Build static artifact mask (tick marks/text) --------
    # Anything that "lights up" in many frames is probably static printing, not bubbles.
    count = np.zeros_like(bg, dtype=np.uint16)
    used = 0
    for p in paths[:min(args.n_bg, len(paths))]:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        d0 = cv2.absdiff(gray, bg)
        d0 = cv2.bitwise_and(d0, roi)
        count += (d0 > args.diff_thr).astype(np.uint16)
        used += 1

    thresh_count = int(args.static_ratio * max(1, used))
    static = (count > thresh_count).astype(np.uint8) * 255
    if args.static_dilate > 0:
        static = cv2.morphologyEx(static, cv2.MORPH_DILATE, k3, iterations=args.static_dilate)

    # -------- Process each frame --------
    for p in paths:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        # absdiff highlights bubbles (moving/shape-changing)
        d = cv2.absdiff(gray, bg)
        d = cv2.bitwise_and(d, roi)

        # threshold diff
        _, m = cv2.threshold(d, args.diff_thr, 255, cv2.THRESH_BINARY)

        # remove static markings
        m = cv2.bitwise_and(m, cv2.bitwise_not(static))

        # clean + merge + fill
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3, iterations=args.open_iters)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5, iterations=args.close_iters)
        m = fill_holes(m)

        # connected components on cleaned mask
        m01 = (m > 127).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m01, connectivity=8)
        out = np.zeros_like(m01, dtype=np.uint8)

        for k in range(1, num):
            x, y, w, h, area = stats[k]
            if area < args.min_area or area > args.max_area:
                continue

            comp = (labels == k).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts or len(cnts[0]) < 20:
                continue

            cnt = cnts[0]

            # Solidity filter: text fragments are often jagged/thin => low solidity
            hull = cv2.convexHull(cnt)
            hull_area = float(cv2.contourArea(hull)) + 1e-6
            solidity = float(area) / hull_area
            if solidity < args.min_solidity:
                continue

            # Ellipticity filter: bubbles are generally oval-ish
            try:
                (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
            except cv2.error:
                continue
            major = max(MA, ma)
            minor = min(MA, ma)
            if major <= 1:
                continue
            ellipticity = minor / major
            if ellipticity < args.min_ellipticity:
                continue

            out[labels == k] = 255

        # final merge to recombine split bubbles (after filtering)
        out_u8 = (out * 255).astype(np.uint8)
        out_u8 = cv2.morphologyEx(out_u8, cv2.MORPH_CLOSE, k5, iterations=args.final_merge_close_iters)
        out_u8 = fill_holes(out_u8)

        cv2.imwrite(str(out_dir / Path(p).name), out_u8)

    print(f"[pseudo] wrote masks -> {out_dir}")

if __name__ == "__main__":
    main()
