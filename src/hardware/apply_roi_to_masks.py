import cv2, glob, argparse
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--roi_mask", required=True)
    ap.add_argument("--border", type=int, default=0)
    args = ap.parse_args()

    roi = cv2.imread(args.roi_mask, cv2.IMREAD_GRAYSCALE)
    if roi is None:
        raise SystemExit("Could not read roi_mask")
    roi = (roi > 127).astype(np.uint8) * 255

    B = args.border
    if B > 0:
        roi[:B,:] = 0; roi[-B:,:] = 0; roi[:,:B] = 0; roi[:,-B:] = 0

    paths = sorted(glob.glob(str(Path(args.masks_dir) / "*.png")))
    for p in paths:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None: 
            continue
        m = ((m > 127).astype(np.uint8) * 255)
        out = cv2.bitwise_and(m, roi)
        cv2.imwrite(p, out)

    print("Applied ROI to", len(paths), "masks")

if __name__ == "__main__":
    main()
