from pathlib import Path
import argparse
import cv2
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--auto_masks_dir", required=True)
    ap.add_argument("--out_masks_dir", required=True)
    ap.add_argument("--brush", type=int, default=10)
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    auto_masks_dir = Path(args.auto_masks_dir)
    out_masks_dir = Path(args.out_masks_dir)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(frames_dir.glob("*.png"))
    if not imgs:
        raise SystemExit(f"No frames in {frames_dir}")

    idx = 0
    brush = args.brush
    mode = None  # "paint" / "erase"
    win = "mask_editor"

    def load_pair(i):
        """
        Returns:
          gray:      HxW uint8
          mask:      HxW uint8 (editable; starts from saved corrected if exists else auto)
          auto_mask: HxW uint8 (auto mask for reset)
        """
        ip = imgs[i]
        gray = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise SystemExit(f"Could not read frame: {ip}")

        mp_auto = auto_masks_dir / ip.name
        mp_out  = out_masks_dir / ip.name

        # auto mask
        if mp_auto.exists():
            auto = cv2.imread(str(mp_auto), cv2.IMREAD_GRAYSCALE)
            if auto is None:
                auto = np.zeros_like(gray)
        else:
            auto = np.zeros_like(gray)
        auto = ((auto > 127).astype(np.uint8) * 255)

        # editable mask: prefer corrected if exists; else start from auto
        if mp_out.exists():
            m = cv2.imread(str(mp_out), cv2.IMREAD_GRAYSCALE)
            if m is None:
                m = auto.copy()
            else:
                m = ((m > 127).astype(np.uint8) * 255)
        else:
            m = auto.copy()

        return gray, m, auto

    gray, mask, auto_mask = load_pair(idx)

    def render():
        # overlay green where mask==1
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        green = np.zeros_like(overlay)
        green[:, :, 1] = 255

        alpha = 0.35
        m3 = cv2.cvtColor((mask > 127).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        overlay = np.where(m3 > 0, (overlay * (1 - alpha) + green * alpha).astype(np.uint8), overlay)

        txt = (
            f"{idx+1}/{len(imgs)}  brush={brush}px  "
            f"s=save  n/p=nav  [ ]=brush  c=clear  r=reset  q=quit"
        )
        cv2.putText(overlay, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow(win, overlay)

    def on_mouse(event, x, y, flags, param):
        nonlocal mask, mode
        if event == cv2.EVENT_LBUTTONDOWN:
            mode = "paint"
        elif event == cv2.EVENT_RBUTTONDOWN:
            mode = "erase"
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            mode = None
        elif event == cv2.EVENT_MOUSEMOVE and mode:
            v = 255 if mode == "paint" else 0
            cv2.circle(mask, (x, y), brush, int(v), thickness=-1)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        render()
        k = cv2.waitKey(30) & 0xFF

        if k == ord("q"):
            break

        if k == ord("s"):
            outp = out_masks_dir / imgs[idx].name
            cv2.imwrite(str(outp), mask)
            print(f"[save] {outp}")

        if k == ord("c"):
            mask[:] = 0
            print("[clear] mask cleared (press 's' to save)")

        if k == ord("r"):
            mask[:] = auto_mask
            print("[reset] restored auto mask (press 's' to save)")

        if k == ord("n"):
            idx = min(len(imgs) - 1, idx + 1)
            gray, mask, auto_mask = load_pair(idx)

        if k == ord("p"):
            idx = max(0, idx - 1)
            gray, mask, auto_mask = load_pair(idx)

        if k == ord("["):
            brush = max(1, brush - 1)

        if k == ord("]"):
            brush = min(200, brush + 1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
