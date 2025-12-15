from pathlib import Path
import argparse
import csv
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


def cc(mask_u8: np.ndarray) -> int:
    m01 = (mask_u8 > 127).astype(np.uint8)
    n, _, _, _ = cv2.connectedComponentsWithStats(m01, connectivity=8)
    return max(0, n - 1)


def load_fp32(ckpt: str):
    m = SmallUNet().cpu()
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    return m


def load_ts(ts: str):
    m = torch.jit.load(ts, map_location="cpu")
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--fp32_ckpt", required=True)
    ap.add_argument("--int8_ts", required=True)
    ap.add_argument("--roi_mask", required=True)
    ap.add_argument("--static_mask", default="")
    ap.add_argument("--num", type=int, default=0, help="0 = all samples")
    ap.add_argument("--out_csv", default="outputs/count_compare.csv")
    ap.add_argument("--out_json", default="outputs/count_compare_summary.json")
    args = ap.parse_args()

    ds = BubbleSegDataset(args.data_root, split=args.split)
    N = len(ds) if args.num == 0 else min(args.num, len(ds))
    print("samples:", N, "of", len(ds))

    roi = cv2.imread(args.roi_mask, cv2.IMREAD_GRAYSCALE)
    if roi is None:
        raise SystemExit(f"Could not read ROI: {args.roi_mask}")

    static = None
    if args.static_mask:
        static = cv2.imread(args.static_mask, cv2.IMREAD_GRAYSCALE)
        if static is None:
            print(f"[warn] Could not read static mask {args.static_mask}; continuing without it")
            static = None

    m_fp32 = load_fp32(args.fp32_ckpt)
    m_int8 = load_ts(args.int8_ts)

    rows = []
    abs_err_fp32 = []
    abs_err_int8 = []

    with torch.no_grad():
        for i in range(N):
            x, y = ds[i]

            gt = (y[0].numpy() > 0.5).astype(np.uint8) * 255
            gt_c = cc(gt)

            # FP32 pred
            probs_fp32 = torch.sigmoid(m_fp32(x.unsqueeze(0)))[0, 0].numpy()
            pred_fp32 = postprocess_bubbles(
                probs=probs_fp32,
                roi_u8=roi,
                thr=THR,
                static_mask_u8=static,
                **POST
            )
            fp32_c = cc(pred_fp32)

            # INT8 pred (TorchScript)
            probs_int8 = torch.sigmoid(m_int8(x.unsqueeze(0)))[0, 0].numpy()
            pred_int8 = postprocess_bubbles(
                probs=probs_int8,
                roi_u8=roi,
                thr=THR,
                static_mask_u8=static,
                **POST
            )
            int8_c = cc(pred_int8)

            abs_err_fp32.append(abs(fp32_c - gt_c))
            abs_err_int8.append(abs(int8_c - gt_c))

            rows.append([i, gt_c, fp32_c, int8_c, abs(fp32_c - gt_c), abs(int8_c - gt_c)])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "gt_count", "fp32_count", "int8_count", "abs_err_fp32", "abs_err_int8"])
        w.writerows(rows)

    def summarize(a):
        a = np.asarray(a, dtype=np.float64)
        return {
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
        }

    summary = {
        "N": N,
        "thr": THR,
        "fp32_ckpt": args.fp32_ckpt,
        "int8_ts": args.int8_ts,
        "abs_err_fp32": summarize(abs_err_fp32),
        "abs_err_int8": summarize(abs_err_int8),
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(__import__("json").dumps(summary, indent=2))

    print("Wrote:", out_csv)
    print("Wrote:", out_json)
    print("FP32 abs error:", summary["abs_err_fp32"])
    print("INT8 abs error:", summary["abs_err_int8"])


if __name__ == "__main__":
    main()
