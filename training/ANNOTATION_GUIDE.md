# Campus Waste – Annotation Guide

Pretrained COCO weights only cover **bottle** and **cup**. Paper, wrappers, food
packaging and general litter need a custom-trained model. This guide explains how
to build that dataset.

## Classes (keep this exact order)

| id | class            | include                                                     | exclude                         |
|----|------------------|-------------------------------------------------------------|---------------------------------|
| 0  | `plastic_bottle` | PET/water/soft-drink bottles, crushed bottles, caps attached | glass bottles held in hand on a table (still label, but see "context") |
| 1  | `paper`          | sheets, crumpled paper, tissues, paper cups, receipts       | notebooks/books being used      |
| 2  | `wrapper`        | chips/biscuit/chocolate wrappers, plastic film, sachets     | bags being carried              |
| 3  | `food_packaging` | takeaway boxes, foil trays, juice tetra-packs, plates       | lunch boxes in use              |
| 4  | `other_litter`   | cans, straws, cigarette butts, anything else discarded      | leaves, natural debris          |

## Collecting images

* Use your **own staged photos** of objects (no people needed) on campus floors,
  corridors, canteen and lawn surfaces. Do **not** collect images of real students
  without consent.
* Aim for **≥150 instances per class** to start (≥500 for good results); vary floor
  texture, lighting (morning/evening/tube-light), distance (0.5–6 m) and angle.
* Include ~10 % **background images** (empty floors) to reduce false positives.
* Public datasets that can be merged after remapping class ids: TACO
  (Trash Annotations in Context), Drinking Waste Classification. Check each licence.

## Labelling

1. Tool: Label Studio, CVAT or Roboflow → export **YOLO (txt)** format.
2. Draw a **tight box** around the whole visible object; include crushed/partly
   hidden parts. One box per object.
3. Occluded > 70 % → skip. Tiny (< 10 px) → skip.
4. An object **in a hand** is still labelled with its class — the action-analysis
   stage decides whether it was littered, not the detector.
5. Label every instance in the image; unlabelled objects teach the model they are background.

Each label line: `class_id x_center y_center width height` (all normalised 0–1).

## Build → train → evaluate

```powershell
python -m training.prepare_dataset --src D:\waste_annotated --out datasets\waste
python -m training.train --data datasets\waste\data.yaml --epochs 100
python -m training.evaluate --weights models\waste_custom.pt --data datasets\waste\data.yaml --split test
```

`train.py` copies the best weights to `models/waste_custom.pt`; the live system
switches to the custom waste model automatically on the next start.

## Quality checklist

* Per-class instance counts printed by `prepare_dataset` are balanced (no class < 1/3 of the largest).
* Test split contains locations/days **not** seen in training (avoid leakage).
* Report precision, recall, mAP@0.5 and mAP@0.5:0.95 per class from `evaluate.py`.
