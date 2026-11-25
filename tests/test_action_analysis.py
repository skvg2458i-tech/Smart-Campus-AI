"""Unit tests for the temporal rule logic.

These feed hand-built *trajectories* into ActionAnalyzer to check the state
machine. They test the rules only - they are not presented as AI detections.
Real-footage checks are done with ``python cli.py video <staged.mp4>``.
"""
from ai.action_analysis import ActionAnalyzer, EventState
from ai.detector import Detection
from config import ActionConfig

H, W = 720, 1280
FPS = 15


def person(tid, x, y=150, w=160, h=420):
    return Detection("person", "person", 0.9, (x, y, x + w, y + h), tid)


def bottle(tid, cx, cy, s=40, conf=0.7):
    return Detection("waste", "plastic_bottle", conf, (cx - s / 4, cy - s / 2, cx + s / 4, cy + s / 2), tid)


def run(frames, cfg=None):
    a = ActionAnalyzer(cfg or ActionConfig())
    events = []
    for i, dets in enumerate(frames):
        events += a.update(dets, i / FPS, (H, W, 3))
    return events


def test_carry_then_drop_is_suspected_littering():
    frames = []
    px = 100
    for i in range(30):                       # 2 s walking while holding the bottle at hand height
        px += 8
        frames.append([person(1, px), bottle(7, px + 150, 380)])
    by = 380
    for i in range(8):                        # bottle falls while person keeps walking
        px += 8; by += 35
        frames.append([person(1, px), bottle(7, px - 8 * i + 150, by)])
    bx, by = frames[-1][1].center
    for i in range(30):                       # person walks away, bottle rests on the floor
        px += 12
        frames.append([person(1, px), bottle(7, bx, by)])
    ev = run(frames)
    assert len(ev) == 1
    e = ev[0]
    assert e.state == EventState.SUSPECTED_LITTERING
    assert e.person_track_id == 1
    assert 0 < e.association_conf <= 1
    assert any("descended" in r for r in e.reasons)


def test_walking_past_existing_bottle_is_normal():
    frames = []
    for i in range(20):                       # bottle already lying on the floor, nobody near
        frames.append([bottle(3, 700, 640)])
    for i in range(60):                       # person walks right past / over it
        frames.append([person(2, 300 + 12 * i, y=250), bottle(3, 700, 640)])
    assert run(frames) == []


def test_drop_then_object_lost_is_uncertain():
    frames = []
    px = 100
    for i in range(30):
        px += 8
        frames.append([person(1, px), bottle(9, px + 150, 380)])
    by = 380
    for i in range(5):
        px += 8; by += 40
        frames.append([person(1, px), bottle(9, px + 150, by)])
    for i in range(45):                       # bottle leaves the frame (e.g. falls below the webcam view)
        px += 8
        frames.append([person(1, px)])
    ev = run(frames)
    assert [e.state for e in ev] == [EventState.UNCERTAIN]
    assert any("lost" in r for r in ev[0].reasons)


def test_placed_on_table_above_ground_zone_is_uncertain():
    frames = []
    px = 100
    for i in range(30):
        px += 8
        frames.append([person(1, px), bottle(4, px + 150, 300)])
    for i in range(30):                       # bottle stays at table height, person walks away
        px += 15
        frames.append([person(1, px), bottle(4, 100 + 30 * 8 + 150, 300)])
    ev = run(frames)
    assert [e.state for e in ev] == [EventState.UNCERTAIN]
    assert any("ABOVE the ground zone" in r for r in ev[0].reasons)


def test_crowded_drop_is_uncertain_with_lower_association():
    frames = []
    px = 100
    for i in range(30):
        px += 8
        frames.append([person(1, px), person(2, px + 200), bottle(5, px + 150, 380)])
    by = 380
    for i in range(8):
        by += 35
        frames.append([person(1, px), person(2, px + 200), bottle(5, px + 150, by)])
    for i in range(30):
        frames.append([person(1, px), person(2, px + 200), bottle(5, px + 150, by)])
    ev = run(frames)
    assert len(ev) == 1
    assert ev[0].state == EventState.UNCERTAIN
    assert ev[0].association_conf <= 0.5


def test_proximity_alone_never_creates_association():
    # person stands still next to a still bottle for a long time -> no carry, no event
    frames = [[person(1, 400), bottle(6, 560, 420)] for _ in range(90)]
    a = ActionAnalyzer()
    for i, f in enumerate(frames):
        assert a.update(f, i / FPS, (H, W, 3)) == []
    assert a.objects[6].carried is False
