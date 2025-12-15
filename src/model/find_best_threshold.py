from pathlib import Path
import argparse
import numpy as np
import torch
import sys

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from small_unet import SmallUNet
from cnn_dataset import BubbleSegDataset

def dice_iou(pred, target, eps=1e-6):
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    inter = (pred * target).sum()
    dice = (2*inter + eps) / (pred.sum() + target.sum() + eps)
    union = pred.sum() + target.sum() - inter
    iou = (inter + eps) / (union + eps)
    return dice, iou

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="val", choices=["train","val"])
    ap.add_argument("--tmin", type=float, default=0.05)
    ap.add_argument("--tmax", type=float, default=0.60)
    ap.add_argument("--steps", type=int, default=24)
    args = ap.parse_args()

    ds = BubbleSegDataset(args.data_root, split=args.split)

    m = SmallUNet().cpu()
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    m.eval()

    ths = np.linspace(args.tmin, args.tmax, args.steps)
    best = (-1, None, None)

    with torch.no_grad():
        for th in ths:
            dices, ious = [], []
            for i in range(len(ds)):
                x, y = ds[i]
                logits = m(x.unsqueeze(0))
                probs = torch.sigmoid(logits)[0,0].numpy()
                pred = (probs > th).astype(np.uint8)
                gt = (y[0].numpy() > 0.5).astype(np.uint8)
                d, j = dice_iou(pred, gt)
                dices.append(d); ious.append(j)
            md = float(np.mean(dices))
            mj = float(np.mean(ious))
            print(f"th={th:.3f}  mean_dice={md:.4f}  mean_iou={mj:.4f}")
            if md > best[0]:
                best = (md, float(th), mj)

    print("\nBEST:", "dice=", best[0], "th=", best[1], "iou=", best[2])

if __name__ == "__main__":
    main()
