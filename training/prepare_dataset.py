"""Prepare a YOLO-format waste dataset for training.

Input: a folder of images with YOLO .txt labels next to them (or in a parallel
``labels/`` folder) - e.g. exported from Label Studio / CVAT / Roboflow in
"YOLO" format using the class order in ``CUSTOM_WASTE_CATEGORIES``.

Output: datasets/waste/{images,labels}/{train,val,test} and a data.yaml.

  python -m training.prepare_dataset --src path/to/annotated --out datasets/waste --val 0.15 --test 0.10

The script validates every label (class id range, normalised coords), reports
per-class instance counts and refuses to continue on broken labels.
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

from config import BASE_DIR, CUSTOM_WASTE_CATEGORIES

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_label(img: Path) -> Path | None:
    for cand in (img.with_suffix(".txt"),
                 Path(str(img.parent).replace("images", "labels")) / (img.stem + ".txt"),
                 img.parent.parent / "labels" / (img.stem + ".txt")):
        if cand.exists():
            return cand
    return None


def validate_label(path: Path, n_classes: int) -> tuple[list[int], list[str]]:
    classes, errors = [], []
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            errors.append(f"{path}:{ln} expected 5 values, got {len(parts)}")
            continue
        try:
            c = int(parts[0]); vals = list(map(float, parts[1:]))
        except ValueError:
            errors.append(f"{path}:{ln} non-numeric value"); continue
        if not 0 <= c < n_classes:
            errors.append(f"{path}:{ln} class id {c} out of range 0..{n_classes - 1}")
        if not all(0.0 <= v <= 1.0 for v in vals):
            errors.append(f"{path}:{ln} coordinates must be normalised to 0..1")
        classes.append(c)
    return classes, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", default=BASE_DIR / "datasets" / "waste", type=Path)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-background", action="store_true",
                    help="keep images without a label file as negative (background) samples")
    a = ap.parse_args()

    names = CUSTOM_WASTE_CATEGORIES
    images = sorted(p for p in a.src.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not images:
        raise SystemExit(f"No images found under {a.src}")
    pairs, errors, counts, missing = [], [], Counter(), 0
    for img in images:
        lbl = find_label(img)
        if lbl is None:
            missing += 1
            if a.allow_background:
                pairs.append((img, None))
            continue
        cls, errs = validate_label(lbl, len(names))
        errors += errs
        counts.update(cls)
        pairs.append((img, lbl))
    if errors:
        print("\n".join(errors[:50]))
        raise SystemExit(f"{len(errors)} label errors - fix them before training")
    random.Random(a.seed).shuffle(pairs)
    n = len(pairs); n_test = int(n * a.test); n_val = int(n * a.val)
    splits = {"test": pairs[:n_test], "val": pairs[n_test:n_test + n_val], "train": pairs[n_test + n_val:]}
    for split, items in splits.items():
        (a.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.copy2(img, a.out / "images" / split / img.name)
            dst = a.out / "labels" / split / (img.stem + ".txt")
            if lbl:
                shutil.copy2(lbl, dst)
            else:
                dst.write_text("")
    data = {"path": str(a.out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
            "names": {i: n for i, n in enumerate(names)}}
    (a.out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"images: {n} (missing labels: {missing}) -> train {len(splits['train'])}, "
          f"val {len(splits['val'])}, test {len(splits['test'])}")
    for i, nm in enumerate(names):
        flag = "  <-- needs more examples" if counts[i] < 150 else ""
        print(f"  {nm:16s} {counts[i]:6d} instances{flag}")
    print("data.yaml written to", a.out / "data.yaml")


if __name__ == "__main__":
    main()
