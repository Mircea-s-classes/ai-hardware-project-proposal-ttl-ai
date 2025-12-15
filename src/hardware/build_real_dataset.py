from pathlib import Path
import argparse
import shutil

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", default="data/processed/real_water/frames")
    ap.add_argument("--masks_root", default="data/processed/real_water/masks")  # corrected masks
    ap.add_argument("--out_root", default="data/cnn/real_water")
    args = ap.parse_args()

    frames_root = Path(args.frames_root)
    masks_root = Path(args.masks_root)
    out_root = Path(args.out_root)
    out_img = out_root / "images"
    out_msk = out_root / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)

    clip_dirs = [p for p in frames_root.iterdir() if p.is_dir()]
    if not clip_dirs:
        raise SystemExit(f"No frame clip dirs in {frames_root}")

    copied = 0
    for clip in sorted(clip_dirs):
        mclip = masks_root / clip.name
        if not mclip.exists():
            print(f"[skip] no corrected masks for {clip.name} in {mclip}")
            continue

        for ip in sorted(clip.glob("*.png")):
            mp = mclip / ip.name
            if not mp.exists():
                continue  # only include frames you actually corrected/saved
            # prefix clip to avoid filename collisions
            out_name = f"{clip.name}__{ip.name}"
            shutil.copy2(ip, out_img / out_name)
            shutil.copy2(mp, out_msk / out_name)
            copied += 1

    print(f"[dataset] wrote {copied} pairs -> {out_root}")

if __name__ == "__main__":
    main()
