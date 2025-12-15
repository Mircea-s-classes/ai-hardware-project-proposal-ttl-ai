from pathlib import Path
import argparse
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler


# make sure we can import sibling files in src/model
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from small_unet import SmallUNet
from cnn_dataset import BubbleSegDataset


# -----------------------------
# Helpers
# -----------------------------
def compute_pos_weight(ds, limit=300):
    """
    Estimate pos_weight ~= (neg/pos) based on up to `limit` samples.
    We then cap it to keep training stable (prevents "everything bubble").
    """
    pos = 0.0
    total = 0.0
    n = min(len(ds), limit)
    for i in range(n):
        _, y = ds[i]
        yb = (y > 0.5).float()
        pos += float(yb.sum().item())
        total += float(yb.numel())
    pos_frac = max(pos / max(total, 1.0), 1e-6)
    neg_frac = 1.0 - pos_frac
    return neg_frac / pos_frac


def soft_dice_loss_from_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (probs * target).sum(dim=1)
    denom = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def dice_iou(probs, target, eps=1e-6):
    """
    probs:  Bx1xHxW in [0,1]
    target: Bx1xHxW in {0,1}
    """
    pred = (probs > 0.5).float()
    target = (target > 0.5).float()

    inter = (pred * target).sum(dim=(1, 2, 3))
    pred_sum = pred.sum(dim=(1, 2, 3))
    targ_sum = target.sum(dim=(1, 2, 3))

    dice = (2 * inter + eps) / (pred_sum + targ_sum + eps)
    union = pred_sum + targ_sum - inter
    iou = (inter + eps) / (union + eps)

    return dice.mean().item(), iou.mean().item()


def mask_border_zero(y, border_px: int):
    """
    Zero out an outer border region in the mask to prevent border artifacts
    from dominating training (common with padding artifacts).
    y: Bx1xHxW
    """
    if border_px <= 0:
        return y
    y2 = y.clone()
    b = border_px
    y2[:, :, :b, :] = 0
    y2[:, :, -b:, :] = 0
    y2[:, :, :, :b] = 0
    y2[:, :, :, -b:] = 0
    return y2


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/cnn/real_water")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--out_ckpt", default="model/real_water_unet.pt")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "dml"])


    # Stability knobs
    ap.add_argument("--border_ignore", type=int, default=8, help="ignore this many pixels on each edge in the loss")
    ap.add_argument("--pos_weight_cap", type=float, default=25.0, help="cap pos_weight to avoid 'all positive' collapse")
    ap.add_argument("--sampler_boost", type=float, default=50.0, help="how strongly to oversample bubble-rich frames")
    ap.add_argument("--use_sampler", action="store_true", help="enable WeightedRandomSampler oversampling")
    ap.add_argument("--dice_weight", type=float, default=0.3, help="weight of dice loss term")
    ap.add_argument("--dice_warmup_epochs", type=int, default=3, help="dice weight = 0 for first N epochs")

    args = ap.parse_args()

    # ... after args = ap.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "dml":
        import torch_directml
        device = torch_directml.device()
    else:
        # auto: try DirectML first, else CPU
        try:
            import torch_directml
            device = torch_directml.device()
        except Exception:
            device = torch.device("cpu")

    print("Using device:", device)


    # Datasets
    train_ds = BubbleSegDataset(args.data_root, split="train")
    val_ds = BubbleSegDataset(args.data_root, split="val")
    print("train samples:", len(train_ds))
    print("val samples:", len(val_ds))

    # pos_weight (capped for stability)
    pw = compute_pos_weight(train_ds, limit=300)
    pw_capped = min(pw, args.pos_weight_cap)
    print("estimated pos_weight:", pw, " -> using capped:", pw_capped)
    pos_weight = torch.tensor([pw_capped], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # DataLoaders
    if args.use_sampler:
        weights = []
        for i in range(len(train_ds)):
            _, y = train_ds[i]
            cov = float((y > 0.5).float().mean().item())  # 0..1
            weights.append(1.0 + args.sampler_boost * cov)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_ld = DataLoader(
            train_ds,
            batch_size=args.batch,
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        print(f"Using sampler: boost={args.sampler_boost}")
    else:
        train_ld = DataLoader(
            train_ds,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
        )
        print("Using shuffle=True (no sampler)")

    val_ld = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # Model
    model = SmallUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_dice = -1.0
    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss = 0.0

        # ramp dice in after warmup (prevents oscillating to all-green early)
        dice_w = 0.0 if ep <= args.dice_warmup_epochs else args.dice_weight

        for x, y in train_ld:
            x = x.to(device)
            y = (y > 0.5).float().to(device)
            y = mask_border_zero(y, args.border_ignore)

            logits = model(x)
            loss = bce(logits, y) + dice_w * soft_dice_loss_from_logits(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())

        # Validation
        model.eval()
        vdice, viou = 0.0, 0.0
        vloss = 0.0
        n = 0

        # diagnostics: are we black because pmax<th? or green because coverage huge?
        pmax_epoch = 0.0
        pmean_epoch = 0.0
        cov35_epoch = 0.0
        cov50_epoch = 0.0

        with torch.no_grad():
            for x, y in val_ld:
                x = x.to(device)
                y = (y > 0.5).float().to(device)
                y = mask_border_zero(y, args.border_ignore)

                logits = model(x)
                loss = bce(logits, y) + dice_w * soft_dice_loss_from_logits(logits, y)

                probs = torch.sigmoid(logits)
                d, i = dice_iou(probs, y)

                bs = x.shape[0]
                vloss += float(loss.item()) * bs
                vdice += float(d) * bs
                viou += float(i) * bs
                n += bs

                # epoch diagnostics (accumulate averages)
                pmax_epoch += float(probs.max().item()) * bs
                pmean_epoch += float(probs.mean().item()) * bs
                cov35_epoch += float((probs > 0.35).float().mean().item()) * bs
                cov50_epoch += float((probs > 0.50).float().mean().item()) * bs

        vdice /= max(1, n)
        viou /= max(1, n)
        vloss /= max(1, n)
        pmax_epoch /= max(1, n)
        pmean_epoch /= max(1, n)
        cov35_epoch /= max(1, n)
        cov50_epoch /= max(1, n)

        dt = time.time() - t0
        print(
            f"ep {ep:03d}  tr_loss={tr_loss/max(1,len(train_ld)):.4f}  val_loss={vloss:.4f}  "
            f"dice={vdice:.4f}  iou={viou:.4f}  dice_w={dice_w:.2f}  "
            f"pmax={pmax_epoch:.3f} pmean={pmean_epoch:.6f} cov@0.35={cov35_epoch:.4f} cov@0.50={cov50_epoch:.4f}  "
            f"({dt:.1f}s)"
        )

        if vdice > best_dice:
            best_dice = vdice
            torch.save(model.state_dict(), out_ckpt)
            print(f"  [saved] {out_ckpt} (best dice {best_dice:.4f})")


if __name__ == "__main__":
    main()
