"""Evidence capture: screenshot + pre/post-event clip + metadata, stored locally.

Layout:  evidence/<INCIDENT_ID>/snapshot.jpg, clip.mp4, metadata.json
Nothing is uploaded anywhere.
"""
from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from config import EVIDENCE_DIR, EvidenceConfig, settings
from services.logger import get_logger

log = get_logger("evidence")


def write_clip(frames: list[np.ndarray], path: Path, fps: float) -> Path:
    """Write an H.264 MP4 (browser-playable) via bundled ffmpeg; fall back to OpenCV mp4v."""
    fps = float(max(2.0, min(fps, 30.0)))
    h, w = frames[0].shape[:2]
    w2, h2 = w - (w % 2), h - (h % 2)  # yuv420p needs even dims
    try:
        import imageio_ffmpeg

        gen = imageio_ffmpeg.write_frames(str(path), (w2, h2), fps=fps, codec="libx264",
                                          pix_fmt_out="yuv420p", macro_block_size=1,
                                          output_params=["-movflags", "+faststart"], quality=None,
                                          ffmpeg_log_level="error")
        gen.send(None)
        for f in frames:
            gen.send(np.ascontiguousarray(cv2.cvtColor(f[:h2, :w2], cv2.COLOR_BGR2RGB)))
        gen.close()
        return path
    except Exception as exc:  # pragma: no cover - fallback path
        log.warning("ffmpeg H.264 writer failed (%s); falling back to OpenCV mp4v", exc)
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        return path


@dataclass
class _PendingClip:
    incident_id: str
    folder: Path
    frames: list
    frames_needed: int
    fps: float
    on_done: object = None
    post: list = field(default_factory=list)


class EvidenceRecorder:
    """Keeps a ring buffer of recent annotated frames and finalises clips after the event."""

    def __init__(self, cfg: EvidenceConfig | None = None, root: Path | None = None):
        self.cfg = cfg or settings.evidence
        self.root = Path(root or EVIDENCE_DIR)
        self.buffer: deque = deque(maxlen=300)
        self.pending: list[_PendingClip] = []
        self._threads: list[threading.Thread] = []

    def push(self, frame: np.ndarray, fps: float) -> None:
        self.buffer.append(frame)
        maxlen = int(max(10, fps) * self.cfg.pre_seconds) + 1
        while len(self.buffer) > maxlen:
            self.buffer.popleft()
        for p in list(self.pending):
            p.post.append(frame)
            if len(p.post) >= p.frames_needed:
                self.pending.remove(p)
                self._finalise(p)

    def capture(self, incident_id: str, snapshot: np.ndarray, raw_frame: np.ndarray, metadata: dict,
                fps: float, on_clip_done=None) -> dict:
        folder = self.root / incident_id
        folder.mkdir(parents=True, exist_ok=True)
        snap_path = folder / "snapshot.jpg"
        cv2.imwrite(str(snap_path), snapshot, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        cv2.imwrite(str(folder / "frame_raw.jpg"), raw_frame, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        meta_path = folder / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        fps = max(2.0, fps)
        self.pending.append(_PendingClip(incident_id, folder, list(self.buffer),
                                         int(fps * self.cfg.post_seconds), fps, on_clip_done))
        log.info("Evidence snapshot saved for %s", incident_id)
        return {"screenshot_path": str(snap_path), "metadata_path": str(meta_path),
                "clip_path": str(folder / "clip.mp4")}

    def _finalise(self, p: _PendingClip) -> None:
        def work():
            frames = p.frames + p.post
            path = write_clip(frames, p.folder / "clip.mp4", p.fps)
            log.info("Evidence clip written for %s (%d frames, %.1f fps)", p.incident_id, len(frames), p.fps)
            if p.on_done:
                p.on_done(p.incident_id, str(path))
        th = threading.Thread(target=work, daemon=True)
        th.start()
        self._threads.append(th)

    def flush(self) -> None:
        """Finish any clip still waiting for post-event frames (e.g. camera stopped)."""
        for p in list(self.pending):
            self.pending.remove(p)
            if p.frames or p.post:
                self._finalise(p)
        for th in self._threads:
            th.join(timeout=30)
        self._threads.clear()
