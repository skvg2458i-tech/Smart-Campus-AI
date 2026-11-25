"""Application-wide logging: console + rotating file in logs/."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config import LOGS_DIR, ensure_dirs

_CONFIGURED = False


def get_logger(name: str = "scai") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        ensure_dirs()
        root = logging.getLogger("scai")
        if root.handlers:  # already configured (e.g. module re-imported by Streamlit)
            _CONFIGURED = True
            return logging.getLogger(name if name.startswith("scai") else f"scai.{name}")
        root.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        fh = RotatingFileHandler(LOGS_DIR / "smart_campus.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(fh)
        root.addHandler(sh)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("scai") else f"scai.{name}")
