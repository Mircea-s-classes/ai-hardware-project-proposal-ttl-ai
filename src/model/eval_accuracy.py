from pathlib import Path
import argparse
import json
import numpy as np
import cv2
import torch
import sys

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from small_unet import SmallUNet
from cnn_dataset import BubbleSegDataset
from postprocess_bubbles import postprocess_bubbles
from bubble_params import THR, POST


def dice_iou(pred_u8, gt_u8, eps=1e-6):
    p = (pred_u8 > 127).astype(np.float32)
    g = (gt_u8 > 127).astype(np.float32)
    inter = (p * g).sum()
    dice = (2.0 * inter + eps) / (p.sum() + g.sum() + eps)
    union = p.sum() + g.sum() - inter
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)


def load_model_fp32(ckpt_path: str):
    m = SmallUNet().cpu()
    sd = torch.load(ckpt_path, map_location="cpu")
    m.load_state_dict(sd)
    m.eval()
    return m


def load_model_ts(ts_path: str):
    m = torch.jit.load(ts_path, map_location="cpu")
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--ckpt", default="", help="FP32 state_dict checkpoint (.pt)")
    ap.add_argument("--ts", default="", help="TorchScript model (.pt), e.g. INT8")
    ap.add_argument("--roi_mask", required=True)
    ap.add_argument("--static_mask", default="")
    ap.add_argument("--num", type=int, default=0, help="0 means all samples")
    ap.add_argument("--out_json", default="outputs/accuracy.json")
    args = ap.parse_args()

    if bool(args.ckpt) == bool(args.ts):
        raise SystemExit("Provide exactly one of --ckpt OR --ts.")

    ds = BubbleSegDataset(args.data_root, split=args.split)
    n = len(ds) if args.num == 0 else min(args.num, len(ds))
    print("eval samples:", n, "of", len(ds))

    roi = cv2.imread(args.roi_mask, cv2.IMREAD_GRAYSCALE)
    if roi is None:
        raise SystemExit(f"Could not read ROI: {args.roi_mask}")

    static = None
    if args.static_mask:
        static = cv2.imread(args.static_mask, cv2.IMREAD_GRAYSCALE)
        if static is None:
            print(f"[warn] Could not read static mask {args.static_mask}; continuing without it")
            static = None

    model = load_model_ts(args.ts) if args.ts else load_model_fp32(args.ckpt)

    dices_raw = []
    ious_raw = []
    dices_post = []
    ious_post = []
    count_absdiff = []

    with torch.no_grad():
        for i in range(n):
            x, y = ds[i]
            logits = model(x.unsqueeze(0))
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

            gt = (y[0].numpy() > 0.5).astype(np.uint8) * 255

            # RAW thresholded mask (no postprocess): isolates model-only quality
            raw = (probs > THR).astype(np.uint8) * 255
            raw = cv2.bitwise_and(raw, (roi > 127).astype(np.uint8) * 255)
            d, j = dice_iou(raw, gt)
            dices_raw.append(d); ious_raw.append(j)

            # POSTPROCESSED mask (your real pipeline): system-level quality
            pred = postprocess_bubbles(
                probs=probs,
                roi_u8=roi,
                thr=THR,
                static_mask_u8=static,
                **POST
            )
            d2, j2 = dice_iou(pred, gt)
            dices_post.append(d2); ious_post.append(j2)

            # output consistency proxy: bubble counts
            def cc(m):
                m01 = (m > 127).astype(np.uint8)
                num, _, _, _ = cv2.connectedComponentsWithStats(m01, connectivity=8)
                return max(0, num - 1)

            count_absdiff.append(abs(cc(pred) - cc(gt)))

    def summarize(arr):
        arr = np.asarray(arr, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    res = {
        "model": args.ts if args.ts else args.ckpt,
        "data_root": args.data_root,
        "split": args.split,
        "n": n,
        "thr": THR,
        "postprocess": POST,
        "dice_raw": summarize(dices_raw),
        "iou_raw": summarize(ious_raw),
        "dice_post": summarize(dices_post),
        "iou_post": summarize(ious_post),
        "bubble_count_abs_error": summarize(count_absdiff),
    }

    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=2))
    print("Wrote:", outp)
    print("RAW  dice mean:", res["dice_raw"]["mean"], "iou mean:", res["iou_raw"]["mean"])
    print("POST dice mean:", res["dice_post"]["mean"], "iou mean:", res["iou_post"]["mean"])

if __name__ == "__main__":
    main()
