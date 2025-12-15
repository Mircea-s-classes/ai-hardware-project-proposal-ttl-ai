from pathlib import Path
import argparse
import cv2
import numpy as np
import torch
import sys

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from small_unet import SmallUNet

def overlay_mask(gray_u8, mask_u8, alpha=0.35):
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    green = np.zeros_like(base)
    green[:, :, 1] = 255
    m3 = cv2.cvtColor((mask_u8 > 127).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    out = np.where(m3 > 0, (base * (1 - alpha) + green * alpha).astype(np.uint8), base)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="model/real_water_unet.pt")
    ap.add_argument("--images_dir", default="data/cnn/real_water/images")
    ap.add_argument("--masks_dir", default="data/cnn/real_water/masks")
    ap.add_argument("--out_dir", default="outputs/preds_sanity")
    ap.add_argument("--num", type=int, default=25)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(Path(args.images_dir).glob("*.png"))
    if not img_paths:
        raise SystemExit(f"No images found in {args.images_dir}")

    model = SmallUNet().cpu()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    for i, ip in enumerate(img_paths[:args.num]):
        base = ip.stem
        mp = Path(args.masks_dir) / ip.name
        if not mp.exists():
            continue

        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        gt  = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)

        if img is None or gt is None:
            continue

        img_r = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)
        gt_r  = cv2.resize(gt,  (args.size, args.size), interpolation=cv2.INTER_NEAREST)

        # model input expects 3ch float [0..1]
        rgb = cv2.cvtColor(img_r, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)[0, 0].numpy()
        pred = (probs > args.threshold).astype(np.uint8) * 255

        cv2.imwrite(str(out_dir / f"{i:03d}_{base}_img.png"), img_r)
        cv2.imwrite(str(out_dir / f"{i:03d}_{base}_gt.png"), gt_r)
        cv2.imwrite(str(out_dir / f"{i:03d}_{base}_pred.png"), pred)
        cv2.imwrite(str(out_dir / f"{i:03d}_{base}_overlay_gt.png"), overlay_mask(img_r, gt_r))
        cv2.imwrite(str(out_dir / f"{i:03d}_{base}_overlay_pred.png"), overlay_mask(img_r, pred))

    print(f"Wrote sanity images to: {out_dir}")

if __name__ == "__main__":
    main()
