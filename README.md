# Smart Campus AI — Waste Detection & Violation Management

AIML college project (BNMIT, Bengaluru). A laptop-webcam system that detects people and
waste objects, tracks them over time, flags *possible* littering actions, saves local
evidence, and lets an **administrator** verify incidents and issue **demonstration** fines
to **fictional** students, who can view them in a student portal.

> **DEMONSTRATION ONLY.** All students are fictional, no real fines are issued, no facial
> recognition is used, and webcam evidence never leaves the laptop.

---

## 1. Features

| Area | What it does |
|---|---|
| Live detection | OpenCV webcam (index 0) → Ultralytics **YOLO11n** (COCO) person detection, boxes, confidences, FPS, status |
| Tracking | **ByteTrack** (Ultralytics built-in) gives temporary, anonymous track IDs to people and objects |
| Waste detection | COCO `bottle` → `plastic_bottle`, COCO `cup` → `cup`. Nothing else from COCO is ever called waste. Custom weights (`models/waste_custom.pt`) for paper / wrapper / food packaging / other litter are loaded automatically once trained |
| Action analysis | Temporal rule baseline over trajectories: carried → separated/descending → stationary on ground. States: `NORMAL`, `POSSIBLE_DROP`, `SUSPECTED_LITTERING`, `UNCERTAIN`, `AWAITING_VERIFICATION` |
| Evidence | Per incident: annotated screenshot, raw frame, **H.264 clip with ~3 s before + ~3 s after**, `metadata.json` (detections, trajectory, thresholds) in `evidence/<INCIDENT_ID>/` |
| Database | SQLite with FKs, CHECK constraints and **triggers**: fines only for CONFIRMED incidents, one fine per incident, fined incidents locked, audit log |
| Admin console | Sign-in, KPIs + Plotly charts, live monitoring, incident review with evidence playback, confirm/reject, fictional-student assignment, fine amount, PDF receipt |
| Student portal | Demo sign-in (USN + PIN), profile, fine history, incident details, PDF download, simulated payment |
| Receipts | ReportLab PDF, watermarked **DEMONSTRATION ONLY** |

## 2. Project structure

```
app.py                  Streamlit entry (navigation: Home / Admin / Student)
config.py               All settings & thresholds (env overrides: SCAI_*)
cli.py                  CLI: image / webcam / video (staged-video testing)
ai/detector.py          YOLO + ByteTrack wrapper (person + waste, custom-weights switch)
ai/action_analysis.py   Temporal littering-action state machine
ai/annotate.py          Boxes, IDs, confidences, states, FPS HUD
camera/stream.py        Webcam open/read/release with error handling
services/pipeline.py    detect → track → analyse → evidence → incident; background MonitorService
services/evidence.py    Ring buffer, screenshots, pre/post clips (bundled ffmpeg, H.264)
services/incidents.py   Auth, incidents, review, fines, access control, stats
services/receipts.py    PDF receipts
services/logger.py      Console + rotating file log (logs/)
database/db.py          Schema, triggers, hashing helpers
database/seed.py        Demo admin + fictional students
dashboards/             home.py, admin.py, student.py, theme.py
training/               prepare_dataset.py, train.py, evaluate.py, train_config.yaml, ANNOTATION_GUIDE.md
tests/                  pytest: action rules, DB/workflow/receipts/evidence, dashboards (AppTest)
evidence/ receipts/ models/ data/ logs/   local runtime folders (git-ignored)
.claude/skills/         Claude Code skills (security-audit, ponytail, mattpocock/skills) – see SOURCES.txt
```

## 3. Installation (Windows, PowerShell)

Requires Python **3.11** (the project was built and tested with 3.11.16). A webcam is needed for live mode.

```powershell
cd "C:\Users\<you>\Desktop\smart ai detection"
py -3.11 -m venv venv            # or: uv venv venv --python 3.11
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m database.seed          # creates data/smart_campus.db + fictional students
```

The YOLO weights (`yolo11n.pt`, ~5 MB) download automatically into `models/` on first run.
Torch installs as CPU-only by default; with an NVIDIA GPU install the CUDA wheel from
pytorch.org and set `SCAI_DEVICE=0`.

## 4. Running

```powershell
streamlit run app.py             # opens http://localhost:8501 (bound to localhost only)
```

Demo credentials (local only, stored hashed):

| Role | Login | Secret |
|---|---|---|
| Administrator | `admin` | `admin@123` |
| Any fictional student | USN e.g. `1XX22AI001` | PIN `1234` |

Override with `SCAI_ADMIN_PASSWORD` / `SCAI_STUDENT_PIN` **before** first seeding.

Other tools:

```powershell
python cli.py webcam                       # OpenCV window, press q to quit
python cli.py image path\to\photo.jpg --save out.jpg
python cli.py video staged_drop.mp4 --out annotated.mp4   # full pipeline on a recorded video
python -m pytest -q                        # automated tests
```

## 5. AI architecture

```
webcam frame ─► YOLO11n (COCO) ──► person boxes ─┐
             └► [optional] custom waste YOLO ──► waste boxes ─┤
                                                 ▼
                                    ByteTrack (persist=True) → track IDs
                                                 ▼
                         ActionAnalyzer (per-track temporal state machine)
                         ├─ carry: object near person AND moved ≥ 0.12×body height with them for ≥ 0.6 s
                         ├─ possible drop: gap ≥ 0.35×body height, or fall ≥ 0.15×body height while person didn't
                         ├─ suspected: object then still ≥ 1 s inside the ground zone (bottom 45 % of frame)
                         └─ uncertain: object lost, came to rest above ground zone, or other people nearby
                                                 ▼
                    EvidenceRecorder (ring buffer → snapshot + pre/post H.264 clip + metadata)
                                                 ▼
                    SQLite incident (AWAITING_VERIFICATION) → admin confirm/reject → demo fine → PDF
```

Design decisions

* **Proximity is never enough.** A person is only recorded as a *possible association* if the
  object was first seen moving *with* that person. A bottle already lying on the floor never
  becomes "carried", so walking past it produces no event (covered by a unit test).
* **Track IDs are anonymous** numbers that reset each session; they are not identities.
* **Uncertain cases are labelled `UNCERTAIN`**, with the reasons stored, and still need review.
* **Track re-identification**: a falling object often gets a new ByteTrack ID; a recently lost
  carried object of the same class nearby hands its history to the new track.
* All thresholds are in `config.ActionConfig`.

## 6. Evaluation metrics

| Component | Metric | How to obtain |
|---|---|---|
| Detector (custom waste) | Precision, Recall, mAP@0.5, mAP@0.5:0.95 per class | `python -m training.evaluate --weights models\waste_custom.pt --data datasets\waste\data.yaml --split test` |
| Tracker | ID switches, MOTA/IDF1 (on annotated clips) | annotate a few staged clips with per-frame IDs; compare with `cli.py video` output |
| Action analysis | Event-level precision / recall / F1, false-alarm rate per hour, uncertain rate | record N staged *drop* videos and N *walk-past* videos; run `cli.py video` on each; count events vs ground truth |
| System | FPS, end-to-end latency, clip-ready time | shown live in the dashboard; measured: ~10–14 FPS for YOLO11n at 640 px on the development laptop (CPU only, Intel Iris Xe graphics, no CUDA) |

## 7. Limitations (be honest in the viva)

* COCO only provides `bottle` and `cup`; **paper, wrappers, food packaging and generic litter
  are not detected** until a custom model is trained (pipeline provided, no dataset included).
* The action module is a **rule-based baseline**, not a learned action-recognition model. It can
  miss drops (object hidden in the hand, falls out of view, very fast motion at ~10 FPS) and can
  flag placing an object down as littering — which is why every incident needs human review.
* The "ground zone" is a fixed fraction of the frame, not a calibrated floor plane.
* A laptop webcam has a narrow field of view and low height; objects often fall out of view
  (→ `UNCERTAIN`).
* ByteTrack IDs break on long occlusions; association confidence drops in crowds.
* Single camera, CPU-only, local SQLite – a demonstration, not a production deployment.

## 8. Future improvements

* Train and publish the custom waste detector (TACO + own campus photos).
* Learned temporal model (e.g. pose keypoints + GRU / video transformer) trained on staged clips.
* Floor-plane calibration (homography) and multi-camera tracking.
* Privacy: automatic face/body blurring in saved evidence; retention policy that auto-deletes evidence.
* Role-based admin accounts with 2FA; HTTPS if ever deployed beyond localhost.

## 9. Ethics & safety

No facial recognition · no real students · no real fines · administrator approval required ·
evidence and recordings stay in local folders (`evidence/`, git-ignored) · receipts watermarked
DEMONSTRATION ONLY · the Streamlit server binds to `localhost` only.
