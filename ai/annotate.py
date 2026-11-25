"""Draw detections, track IDs, confidences, event states and HUD on frames."""
from __future__ import annotations

import cv2
import numpy as np

from ai.detector import Detection

PERSON_COLOR = (235, 160, 40)     # BGR - blue/teal
WASTE_COLOR = (40, 200, 255)      # amber
STATE_COLORS = {
    "NORMAL": (80, 200, 80),
    "POSSIBLE_DROP": (0, 200, 255),
    "SUSPECTED_LITTERING": (40, 40, 230),
    "UNCERTAIN": (200, 120, 200),
    "AWAITING_VERIFICATION": (40, 40, 230),
}


def _label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(th + 6, y)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw(frame: np.ndarray, detections: list[Detection], obj_states: dict | None = None,
         fps: float | None = None, status: str = "", ground_zone: float | None = None) -> np.ndarray:
    img = frame.copy()
    obj_states = obj_states or {}
    h, w = img.shape[:2]
    if ground_zone is not None:
        gy = int(ground_zone * h)
        cv2.line(img, (0, gy), (w, gy), (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(img, "ground zone", (6, gy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox)
        tid = "-" if d.track_id is None else d.track_id
        if d.category == "person":
            color = PERSON_COLOR
            text = f"person #{tid} {d.conf:.2f}"
        else:
            st = obj_states.get(d.track_id, {})
            state = st.get("state", "NORMAL")
            color = STATE_COLORS.get(state, WASTE_COLOR) if state != "NORMAL" else WASTE_COLOR
            text = f"{d.label} #{tid} {d.conf:.2f}"
            if st.get("carried") and state == "NORMAL":
                text += f" | carried by #{st.get('carrier')}"
            elif state != "NORMAL":
                text += f" | {state}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        _label(img, text, x1, y1, color)
    # HUD
    hud = []
    if fps is not None:
        hud.append(f"FPS {fps:4.1f}")
    if status:
        hud.append(status)
    if hud:
        txt = "  |  ".join(hud)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (0, 0), (tw + 16, th + 14), (20, 20, 20), -1)
        cv2.putText(img, txt, (8, th + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img
