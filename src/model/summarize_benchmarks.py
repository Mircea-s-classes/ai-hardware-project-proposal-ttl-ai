import json
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    args = ap.parse_args()

    rows = []
    for p in args.jsons:
        data = json.loads(Path(p).read_text())
        t = data["timing_ms"]["total"]
        rows.append({
            "tag": data.get("tag", Path(p).stem),
            "device": data.get("device", ""),
            "threads": data.get("threads", ""),
            "median_ms": float(t["median"]),
            "p95_ms": float(t["p95"]),
            "fps_est": 1000.0 / float(t["median"]),
            "pre_ms": float(data["timing_ms"]["preprocess"]["median"]),
            "model_ms": float(data["timing_ms"]["model"]["median"]),
            "post_ms": float(data["timing_ms"]["postprocess"]["median"]),
            "count_mean": float(data["bubble_count"]["mean"]),
            "count_med": float(data["bubble_count"]["median"]),
        })

    print("\nTAG\tMED(ms)\tP95(ms)\tFPS\tpre\tmodel\tpost\tcount_mean\tdevice")
    for r in rows:
        print(
            f"{r['tag']}\t"
            f"{r['median_ms']:.2f}\t"
            f"{r['p95_ms']:.2f}\t"
            f"{r['fps_est']:.1f}\t"
            f"{r['pre_ms']:.2f}\t"
            f"{r['model_ms']:.2f}\t"
            f"{r['post_ms']:.2f}\t"
            f"{r['count_mean']:.2f}\t"
            f"{r['device']}"
        )

if __name__ == "__main__":
    main()
