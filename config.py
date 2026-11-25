"""Central configuration for Smart Campus AI.

Every tunable value lives here. Values can be overridden with environment
variables prefixed ``SCAI_`` (for example ``SCAI_CAMERA_INDEX=1``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EVIDENCE_DIR = BASE_DIR / "evidence"
RECEIPTS_DIR = BASE_DIR / "receipts"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DB_PATH = Path(os.getenv("SCAI_DB_PATH", DATA_DIR / "smart_campus.db"))

APP_NAME = "Smart Campus AI"
APP_SUBTITLE = "Waste Detection & Violation Management - BNMIT, Bengaluru (AIML project)"
DEMO_BANNER = "DEMONSTRATION ONLY - fictional students, no real fines"


def _env(name: str, default, cast=str):
    raw = os.getenv(f"SCAI_{name}")
    if raw is None:
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return cast(raw)


@dataclass
class CameraConfig:
    index: int = _env("CAMERA_INDEX", 0, int)
    width: int = _env("CAMERA_WIDTH", 960, int)
    height: int = _env("CAMERA_HEIGHT", 540, int)
    open_timeout_s: float = 6.0
    max_consecutive_failures: int = 30


# COCO class ids that are reasonable *waste candidates*. Only these COCO
# classes are ever labelled as waste - everything else COCO detects is ignored.
# COCO cannot detect paper, wrappers or generic litter; that needs custom weights.
COCO_PERSON_ID = 0
COCO_WASTE_CLASSES: dict[int, str] = {
    39: "plastic_bottle",   # COCO "bottle"
    41: "cup",              # COCO "cup" (disposable cups)
}

# Categories used by the custom waste model (training/ pipeline).
CUSTOM_WASTE_CATEGORIES = ["plastic_bottle", "paper", "wrapper", "food_packaging", "other_litter"]


@dataclass
class ModelConfig:
    # Pretrained COCO model for people (+ bottles/cups). Downloaded automatically by Ultralytics.
    person_model: str = _env("PERSON_MODEL", str(MODELS_DIR / "yolo11n.pt"))
    # Optional custom-trained waste model. When this file exists it is used for
    # waste and the COCO waste classes are switched off.
    custom_waste_model: Path = Path(_env("WASTE_MODEL", str(MODELS_DIR / "waste_custom.pt")))
    tracker: str = "bytetrack.yaml"
    imgsz: int = _env("IMGSZ", 640, int)
    person_conf: float = _env("PERSON_CONF", 0.45, float)
    waste_conf: float = _env("WASTE_CONF", 0.30, float)
    device: str = _env("DEVICE", "cpu")
    iou: float = 0.5


@dataclass
class ActionConfig:
    """Thresholds for the temporal littering-action baseline (see ai/action_analysis.py).

    Distances are normalised by the associated person's bounding-box height so
    they are roughly independent of how far the person is from the camera.
    """
    near_margin: float = 0.10            # expand person box by this fraction when testing "near"
    carry_min_seconds: float = 0.6       # object must move with one person at least this long
    carry_min_motion: float = 0.12       # ...and travel at least this far (x person height)
    separation_thr: float = 0.35         # object-to-person gap that counts as "separated"
    descent_thr: float = 0.15            # downward object movement (x person height)
    descent_window_s: float = 1.0
    stationary_seconds: float = 1.0      # object must stay still this long after the drop
    stationary_tol: float = 0.015        # max movement while "still" (fraction of frame height)
    ground_zone: float = 0.55            # object bottom below this fraction of frame height = ground proxy
    lost_timeout_s: float = 2.0          # possible drop + object lost this long -> uncertain
    crowd_radius: float = 0.6            # other people this close at drop time -> ambiguous association
    report_uncertain: bool = True        # create incident records for UNCERTAIN events too
    track_forget_s: float = 5.0


@dataclass
class EvidenceConfig:
    pre_seconds: float = 3.0
    post_seconds: float = 3.0
    jpeg_quality: int = 90
    cooldown_s: float = 8.0              # minimum time between two incidents


@dataclass
class FineConfig:
    default_amount: float = 200.0
    currency: str = "INR"
    min_amount: float = 50.0
    max_amount: float = 2000.0


@dataclass
class Settings:
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    fine: FineConfig = field(default_factory=FineConfig)


settings = Settings()

# Demo credentials (local only, stored hashed in SQLite at seed time).
DEMO_ADMIN_USERNAME = _env("ADMIN_USER", "admin")
DEMO_ADMIN_PASSWORD = _env("ADMIN_PASSWORD", "admin@123")
DEMO_STUDENT_PIN = _env("STUDENT_PIN", "1234")


def ensure_dirs() -> None:
    for d in (DATA_DIR, EVIDENCE_DIR, RECEIPTS_DIR, MODELS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
