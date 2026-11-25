"""Command-line tools for testing the AI pipeline outside Streamlit.

  python cli.py image <path> [--save out.jpg]        detect people/waste in one image
  python cli.py webcam [--seconds 20] [--no-window]   live webcam with OpenCV window (q to quit)
  python cli.py video <path> [--out annotated.mp4] [--no-incidents]
                                                      run the full pipeline on a staged video file
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import cv2

from ai.annotate import draw
from ai.detector import Detector
from database.db import init_db
from services.evidence import write_clip
from services.pipeline import Pipeline


def cmd_image(a):
    det = Detector()
    img = cv2.imread(a.path)
    if img is None:
        sys.exit(f"Cannot read image {a.path}")
    res = det.detect(img, track=False)
    print(json.dumps({"model": det.describe(), "inference_ms": round(res.inference_ms, 1),
                      "detections": [d.to_dict() for d in res.detections]}, indent=2))
    if a.save:
        cv2.imwrite(a.save, draw(img, res.detections))
        print("saved", a.save)


def cmd_webcam(a):
    from camera.stream import Camera
    init_db()
    pipe = Pipeline(Detector(), create_incidents=not a.no_incidents)
    cam = Camera()
    cam.open()
    pipe.start_session()
    t_end = time.time() + a.seconds if a.seconds else None
    try:
        while t_end is None or time.time() < t_end:
            f = cam.read()
            if f is None:
                continue
            out = pipe.process(f)
            if not a.no_window:
                cv2.imshow("Smart Campus AI - press q to quit", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cam.release()
        pipe.end_session()
        cv2.destroyAllWindows()
    print(json.dumps({"frames": pipe.counters, "unique_person_tracks": len(pipe.person_ids),
                      "unique_waste_tracks": len(pipe.waste_ids), "avg_fps": round(pipe.fps, 1),
                      "events": pipe.events, "incidents": pipe.incident_ids}, indent=2, default=str))


def cmd_video(a):
    init_db()
    cap = cv2.VideoCapture(a.path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video {a.path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    pipe = Pipeline(Detector(), create_incidents=not a.no_incidents)
    pipe.start_session()
    frames_out, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out = pipe.process(f, t=i / src_fps)   # use video time, not wall-clock time
        i += 1
        if a.out:
            frames_out.append(out)
    cap.release()
    pipe.end_session()
    if a.out and frames_out:
        write_clip(frames_out, __import__("pathlib").Path(a.out), src_fps)
    print(json.dumps({"video": a.path, "frames": i, "counters": pipe.counters,
                      "events": pipe.events, "incidents": pipe.incident_ids}, indent=2, default=str))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("image"); p.add_argument("path"); p.add_argument("--save"); p.set_defaults(fn=cmd_image)
    p = sub.add_parser("webcam"); p.add_argument("--seconds", type=float, default=0)
    p.add_argument("--no-window", action="store_true"); p.add_argument("--no-incidents", action="store_true")
    p.set_defaults(fn=cmd_webcam)
    p = sub.add_parser("video"); p.add_argument("path"); p.add_argument("--out")
    p.add_argument("--no-incidents", action="store_true"); p.set_defaults(fn=cmd_video)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
