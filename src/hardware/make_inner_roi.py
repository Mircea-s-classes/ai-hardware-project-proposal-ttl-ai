import argparse
import cv2
import numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi_in", required=True, help="syringe_roi.png")
    ap.add_argument("--roi_out", required=True, help="syringe_roi_inner.png")
    ap.add_argument("--erode", type=int, default=12, help="pixels to erode inward to remove wall markings")
    args = ap.parse_args()

    roi = cv2.imread(args.roi_in, cv2.IMREAD_GRAYSCALE)
    if roi is None:
        raise SystemExit(f"Could not read {args.roi_in}")

    roi = (roi > 127).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*args.erode+1, 2*args.erode+1))
    inner = cv2.erode(roi, k, iterations=1)

    outp = Path(args.roi_out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outp), inner)
    print("Wrote:", outp)

if __name__ == "__main__":
    main()
