import argparse
from pathlib import Path
import glob
import numpy as np
import cv2
import torch
import sys

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from small_unet import SmallUNet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frames_dir", required=True, help="folder of 512x512 frames (png/jpg)")
    ap.add_argument("--roi_mask", required=True, help="syringe_roi.png or syringe_roi_inner.png")
    ap.add_argument("--out_mask", required=True, help="output static mask png")
    ap.add_argument("--n", type=int, default=120, help="how many frames to sample")
    ap.add_argument("--thr", type=float, default=0.20, help="probability threshold used to count 'on' pixels")
    ap.add_argument("--ratio", type=float, default=0.60, help="pixel is static if on in > ratio of frames")
    ap.add_argument("--dilate", type=int, default=2, help="dilate static mask to fully cover markings")
    args = ap.parse_args()

    # load frames
    frames_dir = Path(args.frames_dir)
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        paths += sorted(glob.glob(str(frames_dir / ext)))
    if not paths:
        raise SystemExit(f"No frames found in {frames_dir}")
    paths = paths[: min(args.n, len(paths))]

    # load ROI
    roi = cv2.imread(args.roi_mask, cv2.IMREAD_GRAYSCALE)
    if roi is None:
        raise SystemExit(f"Could not read roi_mask: {args.roi_mask}")
    roi = (roi > 127).astype(np.uint8) * 255

    # load model
    model = SmallUNet().cpu()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    # accumulator
    count = None
    used = 0

    for p in paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue

        # match roi size if needed
        if roi.shape != g.shape:
            roi_use = cv2.resize(roi, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            roi_use = roi

        # model input expects 3ch float in [0,1]
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW

        with torch.no_grad():
            probs = torch.sigmoid(model(x))[0, 0].numpy()

        # count "on" pixels inside ROI
        on = (probs > args.thr).astype(np.uint8) * 255
        on = cv2.bitwise_and(on, roi_use)

        if count is None:
            count = np.zeros_like(on, dtype=np.uint16)
        count += (on > 0).astype(np.uint16)
        used += 1

    if count is None or used < 10:
        raise SystemExit("Too few usable frames to build static mask")

    thresh = int(args.ratio * used)
    static = (count > thresh).astype(np.uint8) * 255

    if args.dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        static = cv2.morphologyEx(static, cv2.MORPH_DILATE, k, iterations=args.dilate)

    outp = Path(args.out_mask)
    outp.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outp), static)

    # debug: save count visualization
    dbg = outp.with_name(outp.stem + "_count.png")
    count_norm = (count.astype(np.float32) / max(1, used) * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(dbg), count_norm)

    print(f"[static] used_frames={used} thr={args.thr} ratio={args.ratio} -> wrote {outp}")
    print(f"[static] wrote debug count heatmap: {dbg}")

if __name__ == "__main__":
    main()
