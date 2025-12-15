import json
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()

    A = json.loads(Path(args.a).read_text())
    B = json.loads(Path(args.b).read_text())

    print("MODEL A:", A["model"])
    print("MODEL B:", B["model"])
    print()

    for k in ["dice_raw", "iou_raw", "dice_post", "iou_post", "bubble_count_abs_error"]:
        print(f"{k}:")
        print(f"  A mean={A[k]['mean']:.4f}  median={A[k]['median']:.4f}  p95={A[k]['p95']:.4f}")
        print(f"  B mean={B[k]['mean']:.4f}  median={B[k]['median']:.4f}  p95={B[k]['p95']:.4f}")
        print()

if __name__ == "__main__":
    main()
