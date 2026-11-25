"""Real YOLO detection + ByteTrack tracking.

* People come from a pretrained COCO model (class 0 "person").
* Waste comes from either
    - a custom-trained waste model (``models/waste_custom.pt``) when present, or
    - the COCO model, restricted to the few COCO classes that are genuine
      waste candidates (bottle, cup - see ``config.COCO_WASTE_CLASSES``).
  COCO has no class for paper, wrappers or generic litter, so those are NOT
  detected until custom weights are trained with ``training/train.py``.

Tracking uses Ultralytics' built-in ByteTrack (``tracker="bytetrack.yaml"``,
``persist=True``) so every box gets a temporary numeric track ID. Track IDs are
anonymous and reset when the tracker is reset - they never identify a person.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import COCO_PERSON_ID, COCO_WASTE_CLASSES, ModelConfig, settings
from services.logger import get_logger

log = get_logger("detector")


@dataclass
class Detection:
    category: str            # "person" or "waste"
    label: str               # e.g. "person", "plastic_bottle", "wrapper"
    conf: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)
    track_id: int | None = None
    source: str = "coco"     # "coco" or "custom"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def to_dict(self) -> dict:
        return {
            "category": self.category, "label": self.label, "conf": round(self.conf, 3),
            "bbox": [round(v, 1) for v in self.bbox], "track_id": self.track_id, "source": self.source,
        }


@dataclass
class FrameResult:
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0

    @property
    def persons(self) -> list[Detection]:
        return [d for d in self.detections if d.category == "person"]

    @property
    def waste(self) -> list[Detection]:
        return [d for d in self.detections if d.category == "waste"]


class Detector:
    """Wraps one or two YOLO models with ByteTrack tracking."""

    def __init__(self, cfg: ModelConfig | None = None):
        from ultralytics import YOLO  # heavy import kept local

        self.cfg = cfg or settings.model
        log.info("Loading person model %s", self.cfg.person_model)
        self.person_model = YOLO(self.cfg.person_model)
        self.waste_model = None
        self.waste_names: dict[int, str] = {}
        custom = Path(self.cfg.custom_waste_model)
        if custom.exists():
            log.info("Loading CUSTOM waste model %s", custom)
            self.waste_model = YOLO(str(custom))
            self.waste_names = dict(self.waste_model.names)
        self.mode = "custom" if self.waste_model is not None else "coco"

    # ------------------------------------------------------------------ info
    def describe(self) -> dict:
        if self.mode == "custom":
            waste = sorted(set(self.waste_names.values()))
        else:
            waste = sorted(set(COCO_WASTE_CLASSES.values()))
        return {
            "person_model": self.cfg.person_model,
            "waste_source": "custom weights" if self.mode == "custom" else "pretrained COCO (limited)",
            "waste_classes": waste,
            "tracker": "ByteTrack",
            "device": self.cfg.device,
        }

    def reset_tracking(self) -> None:
        """Clear ByteTrack state (e.g. when a new camera session starts)."""
        for m in (self.person_model, self.waste_model):
            if m is not None and getattr(m, "predictor", None) is not None:
                m.predictor = None

    # ---------------------------------------------------------------- detect
    def _run(self, model, frame, classes, conf, track: bool):
        kwargs = dict(classes=classes, conf=conf, iou=self.cfg.iou, imgsz=self.cfg.imgsz,
                      device=self.cfg.device, verbose=False)
        if track:
            return model.track(frame, persist=True, tracker=self.cfg.tracker, **kwargs)[0]
        return model.predict(frame, **kwargs)[0]

    @staticmethod
    def _boxes(result):
        b = result.boxes
        if b is None or len(b) == 0:
            return []
        xyxy = b.xyxy.cpu().numpy()
        confs = b.conf.cpu().numpy()
        clss = b.cls.cpu().numpy().astype(int)
        ids = b.id.cpu().numpy().astype(int) if b.id is not None else [None] * len(b)
        return list(zip(xyxy, confs, clss, ids))

    def detect(self, frame: np.ndarray, track: bool = True) -> FrameResult:
        t0 = time.perf_counter()
        out: list[Detection] = []
        coco_classes = [COCO_PERSON_ID]
        if self.mode == "coco":
            coco_classes += list(COCO_WASTE_CLASSES)
        # A single confidence floor is passed to YOLO; per-category floors are applied below.
        floor = min(self.cfg.person_conf, self.cfg.waste_conf)
        res = self._run(self.person_model, frame, coco_classes, floor, track)
        for xyxy, c, k, tid in self._boxes(res):
            if k == COCO_PERSON_ID:
                if c >= self.cfg.person_conf:
                    out.append(Detection("person", "person", float(c), tuple(map(float, xyxy)), tid, "coco"))
            elif k in COCO_WASTE_CLASSES and c >= self.cfg.waste_conf:
                out.append(Detection("waste", COCO_WASTE_CLASSES[k], float(c), tuple(map(float, xyxy)), tid, "coco"))
        if self.waste_model is not None:
            res_w = self._run(self.waste_model, frame, None, self.cfg.waste_conf, track)
            for xyxy, c, k, tid in self._boxes(res_w):
                # offset custom track IDs so they never collide with COCO-model IDs
                out.append(Detection("waste", self.waste_names.get(k, str(k)), float(c),
                                     tuple(map(float, xyxy)), None if tid is None else 100000 + tid, "custom"))
        return FrameResult(out, (time.perf_counter() - t0) * 1000)
