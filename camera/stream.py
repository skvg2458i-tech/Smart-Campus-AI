"""Laptop webcam access through OpenCV (local only - frames never leave this machine)."""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from config import CameraConfig, settings
from services.logger import get_logger

log = get_logger("camera")


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, cfg: CameraConfig | None = None):
        self.cfg = cfg or settings.camera
        self.cap: cv2.VideoCapture | None = None
        self.failures = 0

    def open(self) -> None:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        last_err = "unknown error"
        for be in backends:
            cap = cv2.VideoCapture(self.cfg.index, be)
            if not cap.isOpened():
                cap.release()
                last_err = f"backend {be} could not open camera index {self.cfg.index}"
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            deadline = time.time() + self.cfg.open_timeout_s
            while time.time() < deadline:
                ok, frame = cap.read()
                if ok and frame is not None and frame.size:
                    self.cap = cap
                    self.failures = 0
                    log.info("Camera %d opened (backend %s) at %dx%d", self.cfg.index, be, frame.shape[1], frame.shape[0])
                    return
                time.sleep(0.05)
            cap.release()
            last_err = f"camera index {self.cfg.index} opened but returned no frames"
        raise CameraError(
            f"Cannot access webcam: {last_err}. Check that no other app (Zoom/Teams/Camera) is using it "
            "and that Windows Settings > Privacy & security > Camera allows desktop apps."
        )

    def read(self) -> np.ndarray:
        if self.cap is None:
            raise CameraError("Camera is not open")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.failures += 1
            if self.failures >= self.cfg.max_consecutive_failures:
                raise CameraError("Webcam stopped delivering frames (disconnected or in use by another app)")
            return None
        self.failures = 0
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            log.info("Camera released")

    @property
    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()
