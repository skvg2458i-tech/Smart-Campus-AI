"""Evaluate detection weights on a labelled split and report standard metrics.

  python -m training.evaluate --weights models/waste_custom.pt --data datasets/waste/data.yaml --split test

Reports precision, recall, mAP@0.5 and mAP@0.5:0.95 overall and per class, and
writes reports/eval_<split>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import BASE_DIR, settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    a = ap.parse_args()

    from ultralytics import YOLO

    m = YOLO(a.weights).val(data=a.data, split=a.split, imgsz=a.imgsz, conf=a.conf,
                            device=settings.model.device, plots=True)
    names = m.names
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[int(c)]] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                                     "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4)}
    report = {
        "weights": a.weights, "split": a.split,
        "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
        "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
        "per_class": per_class, "speed_ms": m.speed,
    }
    out = BASE_DIR / "reports"
    out.mkdir(exist_ok=True)
    (out / f"eval_{a.split}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
