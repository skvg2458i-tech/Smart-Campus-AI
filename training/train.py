"""Fine-tune YOLO on the campus waste dataset.

  python -m training.train --data datasets/waste/data.yaml --epochs 100 --model yolo11n.pt

On success the best weights are copied to models/waste_custom.pt, which the
live system picks up automatically (see config.ModelConfig.custom_waste_model).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from config import MODELS_DIR, settings

HERE = Path(__file__).parent


def main():
    cfg = yaml.safe_load((HERE / "train_config.yaml").read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=cfg["model"])
    ap.add_argument("--epochs", type=int, default=cfg["epochs"])
    ap.add_argument("--imgsz", type=int, default=cfg["imgsz"])
    ap.add_argument("--batch", type=int, default=cfg["batch"])
    ap.add_argument("--device", default=settings.model.device)
    ap.add_argument("--project", default=str(Path("runs") / "waste"))
    ap.add_argument("--no-install", action="store_true", help="do not copy best.pt to models/waste_custom.pt")
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.model)
    hyp = {k: v for k, v in cfg.items() if k not in {"model", "epochs", "imgsz", "batch"}}
    results = model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                          project=a.project, name="train", exist_ok=False, **hyp)
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists() and not a.no_install:
        MODELS_DIR.mkdir(exist_ok=True)
        shutil.copy2(best, MODELS_DIR / "waste_custom.pt")
        print("Copied", best, "->", MODELS_DIR / "waste_custom.pt")
        print("Evaluate with: python -m training.evaluate --weights models/waste_custom.pt --data", a.data)


if __name__ == "__main__":
    main()
