"""Temporal littering-action analysis (rule-based baseline).

This is NOT a learned action-recognition model. It is an explainable baseline
that reasons over ByteTrack trajectories of real detections:

1. CARRY      - a waste object moves *together with* one person
                (near that person AND travelling a real distance with them).
                A bottle that was already lying still is never "carried", so a
                person simply walking past it produces no event.
2. POSSIBLE_DROP - a carried object separates from the carrier and/or
                descends quickly while the carrier does not.
3. SUSPECTED_LITTERING - after the possible drop the object becomes
                stationary inside the ground zone, away from the carrier.
4. UNCERTAIN  - a possible drop that cannot be resolved: the object was lost
                (left the frame / occluded), came to rest above the ground
                zone (e.g. placed on a table), or several people were close.
5. AWAITING_VERIFICATION - an incident has been raised; only an administrator
                can confirm or reject it.

The person's temporary track ID is stored as a *possible association* with a
confidence value. Proximity alone never creates an association: the object
must have been observed moving with that person first.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from ai.detector import Detection
from config import ActionConfig, settings


class EventState(str, Enum):
    NORMAL = "NORMAL"
    POSSIBLE_DROP = "POSSIBLE_DROP"
    SUSPECTED_LITTERING = "SUSPECTED_LITTERING"
    UNCERTAIN = "UNCERTAIN"
    AWAITING_VERIFICATION = "AWAITING_VERIFICATION"


@dataclass
class Obs:
    t: float
    cx: float
    cy: float
    bottom: float
    conf: float


@dataclass
class PersonState:
    track_id: int
    bbox: tuple[float, float, float, float]
    last_seen: float
    history: deque = field(default_factory=lambda: deque(maxlen=120))  # (t, cx, cy, h)

    @property
    def height(self) -> float:
        return max(1.0, self.bbox[3] - self.bbox[1])


@dataclass
class ObjectState:
    track_id: int
    label: str
    first_seen: float
    last_seen: float
    history: deque = field(default_factory=lambda: deque(maxlen=300))
    state: EventState = EventState.NORMAL
    carried: bool = False
    carrier_id: int | None = None
    carry_time: dict = field(default_factory=dict)        # person id -> seconds near+co-moving
    carry_start: dict = field(default_factory=dict)       # person id -> (t, cx, cy) when association started
    prior_stationary: bool = False
    drop_t: float | None = None
    drop_reasons: list = field(default_factory=list)
    crowded_at_drop: bool = False
    carrier_h_at_drop: float = 1.0
    emitted: bool = False

    @property
    def last(self) -> Obs:
        return self.history[-1]


@dataclass
class ActionEvent:
    state: EventState
    object_track_id: int
    object_label: str
    person_track_id: int | None
    association_conf: float
    confidence: float
    reasons: list[str]
    t: float
    object_bbox: tuple | None = None

    def to_dict(self) -> dict:
        return {
            "state": self.state.value, "object_track_id": self.object_track_id,
            "object_label": self.object_label, "person_track_id": self.person_track_id,
            "association_conf": round(self.association_conf, 2), "confidence": round(self.confidence, 2),
            "reasons": self.reasons, "t": self.t,
        }


def _dist_to_box(px: float, py: float, box, margin: float) -> float:
    x1, y1, x2, y2 = box
    h = y2 - y1
    w = x2 - x1
    x1 -= w * margin; x2 += w * margin; y1 -= h * margin; y2 += h * margin
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


class ActionAnalyzer:
    def __init__(self, cfg: ActionConfig | None = None):
        self.cfg = cfg or settings.action
        self.persons: dict[int, PersonState] = {}
        self.objects: dict[int, ObjectState] = {}
        self._last_t: float | None = None

    def reset(self) -> None:
        self.persons.clear()
        self.objects.clear()
        self._last_t = None

    # ------------------------------------------------------------------ util
    def _window(self, hist: deque, t_now: float, seconds: float) -> list:
        return [h for h in hist if t_now - h.t <= seconds] if hist and isinstance(hist[0], Obs) else \
               [h for h in hist if t_now - h[0] <= seconds]

    def _is_stationary(self, obj: ObjectState, t_now: float, frame_h: float) -> bool:
        win = self._window(obj.history, t_now, self.cfg.stationary_seconds)
        if len(win) < 3 or (win[-1].t - win[0].t) < self.cfg.stationary_seconds * 0.8:
            return False
        tol = self.cfg.stationary_tol * frame_h
        xs = [o.cx for o in win]; ys = [o.cy for o in win]
        return (max(xs) - min(xs)) <= tol * 2 and (max(ys) - min(ys)) <= tol * 2

    def object_states(self) -> dict[int, dict]:
        """For the overlay: track id -> {state, carried, carrier}."""
        return {tid: {"state": o.state.value, "carried": o.carried, "carrier": o.carrier_id}
                for tid, o in self.objects.items()}

    # ---------------------------------------------------------------- update
    def update(self, detections: list[Detection], t: float, frame_shape) -> list[ActionEvent]:
        """Feed one frame of tracked detections. Returns newly raised events."""
        frame_h = float(frame_shape[0])
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        dt = min(dt, 0.5)  # don't let a stalled frame count as a long carry
        self._last_t = t
        cfg = self.cfg
        events: list[ActionEvent] = []

        # --- persons
        for d in detections:
            if d.category != "person" or d.track_id is None:
                continue
            p = self.persons.get(d.track_id)
            if p is None:
                p = self.persons[d.track_id] = PersonState(d.track_id, d.bbox, t)
            p.bbox, p.last_seen = d.bbox, t
            cx, cy = d.center
            p.history.append((t, cx, cy, p.height))

        seen_objects = set()
        for d in detections:
            if d.category != "waste" or d.track_id is None:
                continue
            seen_objects.add(d.track_id)
            o = self.objects.get(d.track_id)
            if o is None:
                o = self.objects[d.track_id] = ObjectState(d.track_id, d.label, t, t)
                self._inherit_lost_track(o, d, t, frame_h)
            o.last_seen = t
            cx, cy = d.center
            o.history.append(Obs(t, cx, cy, d.bbox[3], d.conf))
            self._update_object(o, t, dt, frame_h, events, d.bbox)

        # --- objects not seen this frame (lost / occluded)
        for tid, o in list(self.objects.items()):
            if tid in seen_objects:
                continue
            if o.state == EventState.POSSIBLE_DROP and not o.emitted and t - o.last_seen > cfg.lost_timeout_s:
                o.state = EventState.UNCERTAIN
                reasons = o.drop_reasons + ["object lost from view after possible drop (left frame or occluded)"]
                events.append(self._event(o, EventState.UNCERTAIN, reasons, t, frame_h))
            if t - o.last_seen > cfg.track_forget_s:
                del self.objects[tid]
        for pid, p in list(self.persons.items()):
            if t - p.last_seen > cfg.track_forget_s:
                del self.persons[pid]
        return events

    def _inherit_lost_track(self, new: ObjectState, d: Detection, t: float, frame_h: float) -> None:
        """A falling object often gets a NEW ByteTrack ID (fast motion / blur).
        Hand the carry history of a recently lost, carried object of the same
        label over to the new track so the drop can still be followed."""
        cx, cy = d.center
        best, best_d = None, 0.5 * frame_h
        for tid, o in self.objects.items():
            if o is new or o.emitted or not o.carried or o.label != d.label:
                continue
            if o.last_seen >= t or t - o.last_seen > self.cfg.lost_timeout_s:
                continue
            dist = math.hypot(cx - o.last.cx, cy - o.last.cy)
            if dist < best_d:
                best, best_d = o, dist
        if best is None:
            return
        for attr in ("carried", "carrier_id", "carry_time", "state", "drop_t", "drop_reasons",
                     "crowded_at_drop", "carrier_h_at_drop", "first_seen"):
            setattr(new, attr, getattr(best, attr))
        new.history.extend(best.history)
        new.drop_reasons = list(best.drop_reasons) + [f"track #{best.track_id} re-identified as #{new.track_id}"]
        del self.objects[best.track_id]

    def _update_object(self, o: ObjectState, t: float, dt: float, frame_h: float, events, bbox) -> None:
        cfg = self.cfg
        if o.emitted:
            return
        cur = o.last
        visible = {pid: p for pid, p in self.persons.items() if p.last_seen == t}

        # Pre-existing stationary object seen before anyone carried it.
        if not o.carried and o.state == EventState.NORMAL and self._is_stationary(o, t, frame_h):
            o.prior_stationary = True

        # ---------------- carry association (needs proximity AND co-movement)
        near = {pid: p for pid, p in visible.items()
                if _dist_to_box(cur.cx, cur.cy, p.bbox, cfg.near_margin) <= 1e-6}
        for pid, p in near.items():
            if pid not in o.carry_start:
                o.carry_start[pid] = (t, cur.cx, cur.cy)
            t0, x0, y0 = o.carry_start[pid]
            moved = math.hypot(cur.cx - x0, cur.cy - y0) / p.height
            # credit time only once the object has actually moved since association started
            if moved >= cfg.carry_min_motion * 0.5:
                o.carry_time[pid] = o.carry_time.get(pid, 0.0) + dt
            if (o.carry_time.get(pid, 0.0) >= cfg.carry_min_seconds and moved >= cfg.carry_min_motion):
                if not o.carried or o.carrier_id != pid:
                    o.carried, o.carrier_id = True, pid
                if o.state in (EventState.POSSIBLE_DROP,):
                    # picked back up -> cancel the possible drop
                    o.state, o.drop_t, o.drop_reasons = EventState.NORMAL, None, []
                o.prior_stationary = False
        for pid in list(o.carry_start):
            if pid not in near:
                o.carry_start.pop(pid, None)

        # ---------------- possible drop
        if o.carried and o.state == EventState.NORMAL:
            carrier = self.persons.get(o.carrier_id)
            if carrier is not None and t - carrier.last_seen > 0.5 and self._is_stationary(o, t, frame_h):
                o.state, o.drop_t = EventState.POSSIBLE_DROP, t
                o.carrier_h_at_drop = carrier.height
                o.drop_reasons = [f"object observed moving with person #{carrier.track_id}",
                                  f"person #{carrier.track_id} left view while the object stayed behind"]
            elif carrier is not None:
                h = carrier.height
                reasons = []
                gap = _dist_to_box(cur.cx, cur.cy, carrier.bbox, 0.0) / h
                if gap >= cfg.separation_thr:
                    reasons.append(f"object separated from person #{carrier.track_id} (gap {gap:.2f} x body height)")
                win = [ob for ob in o.history if t - ob.t <= cfg.descent_window_s]
                if len(win) >= 2:
                    descent = (win[-1].bottom - min(ob.bottom for ob in win)) / h
                    pwin = [ph for ph in carrier.history if t - ph[0] <= cfg.descent_window_s]
                    p_desc = (pwin[-1][2] - min(ph[2] for ph in pwin)) / h if len(pwin) >= 2 else 0.0
                    if descent >= cfg.descent_thr and descent - p_desc >= cfg.descent_thr * 0.7:
                        reasons.append(f"object descended {descent:.2f} x body height while person did not")
                if reasons:
                    o.state = EventState.POSSIBLE_DROP
                    o.drop_t = t
                    o.drop_reasons = ["object observed moving with person "
                                      f"#{carrier.track_id} for {o.carry_time.get(carrier.track_id, 0):.1f}s"] + reasons
                    o.carrier_h_at_drop = h
                    others = [p for pid, p in visible.items() if pid != carrier.track_id and
                              _dist_to_box(cur.cx, cur.cy, p.bbox, 0.0) / p.height <= cfg.crowd_radius]
                    o.crowded_at_drop = bool(others)
                    if others:
                        o.drop_reasons.append(f"{len(others)} other person(s) nearby - association ambiguous")

        # ---------------- resolve a possible drop
        if o.state == EventState.POSSIBLE_DROP and self._is_stationary(o, t, frame_h):
            carrier = self.persons.get(o.carrier_id)
            still_held = carrier is not None and carrier.last_seen == t and \
                _dist_to_box(cur.cx, cur.cy, carrier.bbox, 0.0) <= 1e-6 and \
                not any("descended" in r for r in o.drop_reasons)
            if still_held:
                return  # e.g. person stopped while holding it; keep waiting
            on_ground = cur.bottom >= cfg.ground_zone * frame_h
            reasons = list(o.drop_reasons)
            reasons.append(f"object stationary for >= {cfg.stationary_seconds:.1f}s after separation")
            if on_ground and not o.crowded_at_drop:
                reasons.append("object at rest inside the ground zone")
                o.state = EventState.SUSPECTED_LITTERING
            else:
                if not on_ground:
                    reasons.append("object at rest ABOVE the ground zone (may have been placed on a surface)")
                o.state = EventState.UNCERTAIN
            events.append(self._event(o, o.state, reasons, t, frame_h, bbox))

    def _event(self, o: ObjectState, state: EventState, reasons, t, frame_h, bbox=None) -> ActionEvent:
        total = sum(o.carry_time.values()) or 1.0
        carrier_time = o.carry_time.get(o.carrier_id, 0.0)
        dominance = carrier_time / total
        assoc = dominance * min(1.0, carrier_time / (2 * self.cfg.carry_min_seconds))
        if o.crowded_at_drop:
            assoc *= 0.5
        confs = [ob.conf for ob in o.history]
        det_conf = sum(confs) / len(confs) if confs else 0.0
        evidence_terms = 0.6
        if any("separated" in r for r in reasons):
            evidence_terms += 0.2
        if any("descended" in r for r in reasons):
            evidence_terms += 0.2
        conf = det_conf * evidence_terms
        if state == EventState.UNCERTAIN:
            conf *= 0.6
        o.emitted = True
        o.state = EventState.AWAITING_VERIFICATION if (state == EventState.SUSPECTED_LITTERING or
                                                       self.cfg.report_uncertain) else state
        person = o.carrier_id if assoc > 0 else None
        reasons = list(reasons) + [
            "person track ID is a temporary, anonymous ByteTrack ID - a POSSIBLE association, not identification"]
        return ActionEvent(state, o.track_id, o.label, person, assoc, conf, reasons, t, bbox)
