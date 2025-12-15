from pathlib import Path
import argparse
import cv2
import numpy as np

def flood_fill_holes(mask_u8: np.ndarray) -> np.ndarray:
    h, w = mask_u8.shape
    ff = mask_u8.copy()
    tmp = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, tmp, (0, 0), 255)
    ff_inv = cv2.bitwise_not(ff)
    return mask_u8 | ff_inv

def filled_ellipses_from_edges(gray, canny1=40, canny2=120, min_area=120):
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(g, canny1, canny2)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(gray, dtype=np.uint8)

    for c in cnts:
        if len(c) < 20:
            continue
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        try:
            ell = cv2.fitEllipse(c)
            cv2.ellipse(mask, ell, 255, thickness=-1)  # FILLED ellipse
        except cv2.error:
            pass

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = flood_fill_holes(mask)
    return mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", default="data/processed/real_water/frames")
    ap.add_argument("--out_root", default="data/processed/real_water/masks_auto")
    ap.add_argument("--canny1", type=int, default=40)
    ap.add_argument("--canny2", type=int, default=120)
    ap.add_argument("--min_area", type=int, default=120)
    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    clips = [p for p in frames_root.iterdir() if p.is_dir()]
    if not clips:
        raise SystemExit(f"No clip folders found in {frames_root}")

    for clip in sorted(clips):
        out_clip = out_root / clip.name
        out_clip.mkdir(parents=True, exist_ok=True)

        imgs = sorted(clip.glob("*.png"))
        if not imgs:
            continue

        for ip in imgs:
            gray = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
            mask = filled_ellipses_from_edges(
                gray, canny1=args.canny1, canny2=args.canny2, min_area=args.min_area
            )
            cv2.imwrite(str(out_clip / ip.name), mask)

        print(f"[pseudolabel] {clip.name}: {len(imgs)} masks -> {out_clip}")

if __name__ == "__main__":
    main()
