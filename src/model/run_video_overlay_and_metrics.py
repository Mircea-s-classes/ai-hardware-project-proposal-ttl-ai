from pathlib import Path
import argparse
import time
import json
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import sys

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from small_unet import SmallUNet
from postprocess_bubbles import postprocess_bubbles
from bubble_params import THR, POST


def pick_device(name: str):
    name = name.lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "dml":
        import torch_directml
        return torch_directml.device()
    # auto: prefer DML if available
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


def component_count(mask_u8: np.ndarray) -> int:
    """Counts connected components in a binary-ish mask (0/1 or 0/255)."""
    m01 = (mask_u8 > 0).astype(np.uint8)
    num, _, _, _ = cv2.connectedComponentsWithStats(m01, connectivity=8)
    return max(0, num - 1)


@dataclass
class Track:
    track_id: int
    cx: float
    cy: float
    last_seen: int
    seen_count: int = 1
    confirmed: bool = False


class BubbleCounter:
    """
    Cumulative "unique bubble" counter using centroid tracking on connected components.

    - confirm_frames prevents counting 1-frame flicker
    - max_age prevents immediate re-count when a bubble disappears briefly
    - max_dist controls association between frames (pixels, in the resolution you feed in)
    """
    def __init__(self, min_area=60, max_dist=25.0, max_age=10, confirm_frames=2):
        self.min_area = int(min_area)
        self.max_dist = float(max_dist)
        self.max_age = int(max_age)
        self.confirm_frames = int(confirm_frames)

        self.tracks: list[Track] = []
        self.next_id = 1
        self.total_unique = 0
        self.frame_idx = 0

        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.total_unique = 0
        self.frame_idx = 0

    def _detections_from_mask(self, mask_u8: np.ndarray):
        m = mask_u8
        if m.dtype != np.uint8:
            m = m.astype(np.uint8)

        m = (m > 0).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._kernel, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._kernel, iterations=1)

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)

        dets = []
        for i in range(1, num):  # 0 is background
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            cx, cy = centroids[i]
            dets.append((float(cx), float(cy), area))
        return dets

    def update(self, mask_u8: np.ndarray) -> int:
        """
        Update tracker for one frame.
        Returns current total_unique (cumulative).
        """
        self.frame_idx += 1
        dets = self._detections_from_mask(mask_u8)

        unmatched_tracks = set(range(len(self.tracks)))
        matched_dets = set()

        # Greedy nearest-neighbor assignment: det -> best track
        for di, (dcx, dcy, _area) in enumerate(dets):
            best_ti = None
            best_dist = None

            for ti in list(unmatched_tracks):
                t = self.tracks[ti]
                dist = ((t.cx - dcx) ** 2 + (t.cy - dcy) ** 2) ** 0.5
                if dist <= self.max_dist and (best_dist is None or dist < best_dist):
                    best_dist = dist
                    best_ti = ti

            if best_ti is not None:
                t = self.tracks[best_ti]
                t.cx, t.cy = dcx, dcy
                t.last_seen = self.frame_idx
                t.seen_count += 1

                if (not t.confirmed) and (t.seen_count >= self.confirm_frames):
                    t.confirmed = True
                    self.total_unique += 1

                unmatched_tracks.remove(best_ti)
                matched_dets.add(di)

        # New tracks for unmatched detections
        for di, (dcx, dcy, _area) in enumerate(dets):
            if di in matched_dets:
                continue
            self.tracks.append(
                Track(
                    track_id=self.next_id,
                    cx=dcx,
                    cy=dcy,
                    last_seen=self.frame_idx,
                    seen_count=1,
                    confirmed=False,
                )
            )
            self.next_id += 1

        # Prune old tracks
        kept = []
        for t in self.tracks:
            if (self.frame_idx - t.last_seen) <= self.max_age:
                kept.append(t)
        self.tracks = kept

        return self.total_unique


def load_model_fp32(ckpt_path: str, device):
    model = SmallUNet().to(device)
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model


def load_model_torchscript(ts_path: str):
    # Quantized TorchScript is CPU-oriented; load on CPU
    model = torch.jit.load(ts_path, map_location="cpu")
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="FP32 state_dict checkpoint (.pt)")
    ap.add_argument("--ts", default="", help="TorchScript model (.pt), e.g. INT8 quantized")
    ap.add_argument("--video", required=True)
    ap.add_argument("--roi_mask", required=True)
    ap.add_argument("--static_mask", default="")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "dml"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--num_frames", type=int, default=600)
    ap.add_argument("--every_n", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--model_size", type=int, default=512, help="resize model input to NxN")
    ap.add_argument("--save_overlay", action="store_true")
    ap.add_argument("--overlay_out", default="outputs/overlay.mp4")
    ap.add_argument("--metrics_out", default="outputs/metrics.json")
    ap.add_argument("--tag", default="")

    # NEW: cumulative tracking knobs (all in OUTPUT VIDEO pixel units)
    ap.add_argument("--track_unique", action="store_true", help="Draw cumulative unique-bubble counter on overlay")
    ap.add_argument("--track_min_area", type=int, default=60, help="Min blob area to consider (pixels)")
    ap.add_argument("--track_max_dist", type=float, default=25.0, help="Max centroid move between frames (pixels)")
    ap.add_argument("--track_max_age", type=int, default=10, help="Frames to keep a track alive if unseen")
    ap.add_argument("--track_confirm_frames", type=int, default=2, help="Frames blob must persist before counting unique")

    args = ap.parse_args()

    if bool(args.ckpt) == bool(args.ts):
        raise SystemExit("Provide exactly one of --ckpt (FP32) OR --ts (TorchScript).")

    torch.set_num_threads(args.threads)

    # TorchScript INT8 path => force CPU to keep comparison meaningful
    if args.ts:
        device = torch.device("cpu")
    else:
        device = pick_device(args.device)

    print("Using device:", device)

    roi = cv2.imread(args.roi_mask, cv2.IMREAD_GRAYSCALE)
    if roi is None:
        raise SystemExit(f"Could not read ROI: {args.roi_mask}")

    static = None
    if args.static_mask:
        static = cv2.imread(args.static_mask, cv2.IMREAD_GRAYSCALE)
        if static is None:
            print(f"[warn] Could not read static mask {args.static_mask}; continuing without it")
            static = None

    if args.ts:
        model = load_model_torchscript(args.ts)  # CPU
    else:
        model = load_model_fp32(args.ckpt, device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    vw = None
    if args.save_overlay:
        outp = Path(args.overlay_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(outp), fourcc, fps_in / max(1, args.every_n), (W, H))

    # NEW: tracker (used if track_unique)
    tracker = BubbleCounter(
        min_area=args.track_min_area,
        max_dist=args.track_max_dist,
        max_age=args.track_max_age,
        confirm_frames=args.track_confirm_frames,
    )

    t_pre, t_model, t_post, t_total = [], [], [], []
    counts = []
    unique_total_final = 0

    used = 0
    idx = 0

    while used < args.num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every_n != 0:
            idx += 1
            continue

        t0 = time.perf_counter()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_m = cv2.resize(gray, (args.model_size, args.model_size), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(gray_m, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

        x = x.to(device)
        t1 = time.perf_counter()

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        t2 = time.perf_counter()

        pred = postprocess_bubbles(
            probs=probs,
            roi_u8=roi,
            thr=THR,
            static_mask_u8=static,
            **POST
        )
        t3 = time.perf_counter()

        if vw is not None:
            pred_big = cv2.resize(pred, (W, H), interpolation=cv2.INTER_NEAREST)

            now_count = component_count(pred_big)

            if args.track_unique:
                unique_total = tracker.update(pred_big)
                unique_total_final = unique_total
            else:
                unique_total = unique_total_final

            overlay = frame.copy()
            overlay[pred_big > 0] = (overlay[pred_big > 0] * 0.65 + np.array([0, 255, 0]) * 0.35).astype(np.uint8)

            cv2.putText(
                overlay,
                f"thr={THR:.2f} now={now_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )
            if args.track_unique:
                cv2.putText(
                    overlay,
                    f"unique_total={unique_total}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2
                )

            vw.write(overlay)

        t4 = time.perf_counter()

        if used >= args.warmup:
            t_pre.append((t1 - t0) * 1000.0)
            t_model.append((t2 - t1) * 1000.0)
            t_post.append((t3 - t2) * 1000.0)
            t_total.append((t4 - t0) * 1000.0)
            counts.append(component_count(pred))

        used += 1
        idx += 1

    cap.release()
    if vw is not None:
        vw.release()

    if len(t_total) == 0:
        raise SystemExit("No timing samples collected (video too short or warmup too large).")

    def stats(arr):
        arr = np.array(arr, dtype=np.float64)
        return dict(
            median=float(np.median(arr)),
            p95=float(np.percentile(arr, 95)),
            mean=float(np.mean(arr)),
            min=float(np.min(arr)),
            max=float(np.max(arr)),
        )

    res = {
        "tag": args.tag,
        "device": str(device),
        "threads": args.threads,
        "model": args.ts if args.ts else args.ckpt,
        "video": args.video,
        "model_size": args.model_size,
        "every_n": args.every_n,
        "thr": THR,
        "postprocess": POST,
        "timing_ms": {
            "preprocess": stats(t_pre),
            "model": stats(t_model),
            "postprocess": stats(t_post),
            "total": stats(t_total),
        },
        "bubble_count": {
            "mean": float(np.mean(counts)),
            "median": float(np.median(counts)),
            "min": int(np.min(counts)),
            "max": int(np.max(counts)),
        },
        "unique_bubbles": {
            "enabled": bool(args.track_unique and args.save_overlay),
            "total_unique_final": int(unique_total_final),
            "params": {
                "min_area": args.track_min_area,
                "max_dist": args.track_max_dist,
                "max_age": args.track_max_age,
                "confirm_frames": args.track_confirm_frames,
            } if args.track_unique else {},
        },
        "overlay_saved": bool(args.save_overlay),
        "overlay_out": args.overlay_out if args.save_overlay else "",
    }

    out_json = Path(args.metrics_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=2))
    print("Wrote:", out_json)
    if args.save_overlay:
        print("Overlay:", args.overlay_out)


if __name__ == "__main__":
    main()
