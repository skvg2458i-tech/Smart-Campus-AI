"""End-to-end frame pipeline: detect -> track -> analyse -> evidence -> incident.

``Pipeline`` processes frames from any source (webcam or a video file).
``MonitorService`` runs the pipeline on the webcam in a background thread so the
Streamlit UI can start/stop it and poll the latest annotated frame.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime

import numpy as np

from ai.action_analysis import ActionAnalyzer, EventState
from ai.annotate import draw
from ai.detector import Detector
from camera.stream import Camera, CameraError
from config import settings
from database.db import connect, new_incident_id, now_iso
from services import incidents as inc_svc
from services.evidence import EvidenceRecorder
from services.logger import get_logger

log = get_logger("pipeline")


class Pipeline:
    def __init__(self, detector: Detector, db_path=None, evidence_root=None, create_incidents: bool = True):
        self.detector = detector
        self.analyzer = ActionAnalyzer()
        self.evidence = EvidenceRecorder(root=evidence_root)
        self.db_path = db_path
        self.create_incidents = create_incidents
        self.session_id: int | None = None
        self._fps_times: deque = deque(maxlen=30)
        self.fps = 0.0
        self.counters = {"frames": 0, "person_detections": 0, "waste_detections": 0}
        self.person_ids: set = set()
        self.waste_ids: set = set()
        self.last_incident_t = -1e9
        self.events: list[dict] = []
        self.incident_ids: list[str] = []
        self.status = "Idle"

    # ------------------------------------------------------------- session
    def start_session(self) -> None:
        self.detector.reset_tracking()
        self.analyzer.reset()
        with connect(self.db_path) as conn:
            cur = conn.execute("INSERT INTO monitor_sessions(started_at) VALUES (?)", (now_iso(),))
            self.session_id = cur.lastrowid

    def save_session(self, final: bool = False) -> None:
        if self.session_id is None:
            return
        with connect(self.db_path) as conn:
            conn.execute(
                """UPDATE monitor_sessions SET frames=?, person_detections=?, waste_detections=?,
                   unique_persons=?, unique_waste=?, avg_fps=?, ended_at=COALESCE(?, ended_at) WHERE id=?""",
                (self.counters["frames"], self.counters["person_detections"], self.counters["waste_detections"],
                 len(self.person_ids), len(self.waste_ids), round(self.fps, 2), now_iso() if final else None,
                 self.session_id),
            )

    def end_session(self) -> None:
        self.evidence.flush()
        self.save_session(final=True)

    # --------------------------------------------------------------- frame
    def process(self, frame: np.ndarray, t: float | None = None) -> np.ndarray:
        t = time.time() if t is None else t
        self._fps_times.append(time.perf_counter())
        if len(self._fps_times) >= 2:
            span = self._fps_times[-1] - self._fps_times[0]
            self.fps = (len(self._fps_times) - 1) / span if span > 0 else 0.0

        res = self.detector.detect(frame, track=True)
        persons, waste = res.persons, res.waste
        self.counters["frames"] += 1
        self.counters["person_detections"] += len(persons)
        self.counters["waste_detections"] += len(waste)
        self.person_ids.update(d.track_id for d in persons if d.track_id is not None)
        self.waste_ids.update(d.track_id for d in waste if d.track_id is not None)

        events = self.analyzer.update(res.detections, t, frame.shape)
        states = self.analyzer.object_states()
        active = [s["state"] for s in states.values() if s["state"] != "NORMAL"]
        if active:
            self.status = "ALERT: " + ", ".join(sorted(set(active)))
        elif any(s["carried"] for s in states.values()):
            self.status = "Object carried by person"
        else:
            self.status = f"Monitoring - {len(persons)} person(s), {len(waste)} waste object(s)"

        annotated = draw(frame, res.detections, states, fps=self.fps, status=self.status,
                         ground_zone=settings.action.ground_zone)
        for ev in events:
            self._handle_event(ev, frame, annotated, res, t)
        self.evidence.push(annotated, self.fps or 10.0)
        if self.counters["frames"] % 30 == 0:
            self.save_session()
        return annotated

    def _handle_event(self, ev, raw, annotated, res, t) -> None:
        d = ev.to_dict()
        d["event_time"] = datetime.fromtimestamp(t).isoformat(timespec="seconds")
        self.events.append(d)
        log.info("Action event: %s", d)
        if not self.create_incidents:
            return
        if ev.state == EventState.UNCERTAIN and not settings.action.report_uncertain:
            return
        if t - self.last_incident_t < settings.evidence.cooldown_s:
            log.info("Event suppressed by incident cooldown")
            return
        self.last_incident_t = t
        iid = new_incident_id()
        metadata = {
            "incident_id": iid, "event": d, "session_id": self.session_id,
            "detections": [x.to_dict() for x in res.detections],
            "object_trajectory": [vars(o) for o in self.analyzer.objects[ev.object_track_id].history]
            if ev.object_track_id in self.analyzer.objects else [],
            "model": self.detector.describe(), "thresholds": vars(settings.action),
            "disclaimer": "Person track IDs are temporary and anonymous. No facial recognition is used. "
                          "Association with a person is a possibility that requires administrator verification.",
        }
        evid = self.evidence.capture(
            iid, annotated, raw, metadata, self.fps or 10.0,
            on_clip_done=lambda i, p: inc_svc.set_clip_path(i, p, self.db_path),
        )
        inc_svc.create_incident(d, evid, self.session_id, self.db_path, incident_id=iid)
        self.incident_ids.append(iid)


class MonitorService:
    """Background webcam monitoring controlled from the Streamlit UI."""

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.error: str | None = None
        self.running = False
        self.loading = False
        self.detector: Detector | None = None
        self.pipeline: Pipeline | None = None

    def start(self) -> None:
        if self.running or self.loading:
            return
        log.info("Monitoring START requested")
        self._stop.clear()
        self.error = None
        self.loading = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="scai-monitor")
        self._thread.start()

    def stop(self) -> None:
        log.info("Monitoring STOP requested")
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)
        self.running = False
        self.loading = False

    def snapshot(self) -> dict:
        p = self.pipeline
        return {
            "running": self.running, "loading": self.loading, "error": self.error,
            "fps": round(p.fps, 1) if p else 0.0, "status": p.status if p else "Idle",
            "counters": dict(p.counters) if p else {}, "unique_persons": len(p.person_ids) if p else 0,
            "unique_waste": len(p.waste_ids) if p else 0, "events": list(p.events[-10:]) if p else [],
            "incidents": list(p.incident_ids) if p else [],
            "model": self.detector.describe() if self.detector else None,
        }

    def _run(self) -> None:
        import cv2

        cam = Camera()
        try:
            if self.detector is None:
                self.detector = Detector()
            cam.open()
            self.pipeline = Pipeline(self.detector)
            self.pipeline.start_session()
            self.running, self.loading = True, False
            while not self._stop.is_set():
                frame = cam.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                out = self.pipeline.process(frame)
                ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self.latest_jpeg = buf.tobytes()
        except CameraError as exc:
            self.error = str(exc)
            log.error("Camera error: %s", exc)
        except Exception as exc:  # keep the UI alive and show the error
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("Monitor crashed")
        finally:
            cam.release()
            if self.pipeline:
                self.pipeline.end_session()
            self.running, self.loading = False, False
