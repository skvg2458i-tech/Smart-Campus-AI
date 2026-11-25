# Project Demonstration Guide (≈10 minutes)

## Before the demo
1. Close Zoom/Teams/Camera app (they lock the webcam).
2. `.\venv\Scripts\Activate.ps1` → `python -m pytest -q` (show all tests pass).
3. `streamlit run app.py` → browser opens `http://localhost:8501`.
4. Props: one empty **plastic bottle**. Place the laptop so the camera sees you from the waist
   up **and** some floor/desk area in the lower part of the frame. Good, even lighting.

## Script

| # | Step | What to say / show |
|---|---|---|
| 1 | Home page | Purpose, 4-stage pipeline, ethics box (no face recognition, fictional students, DEMO receipts) |
| 2 | Admin Console → sign in (`admin` / `admin@123`) | Access control: nothing visible before sign-in |
| 3 | Live Monitoring → **Start camera** | Person box + `person #ID conf`, FPS, status line. Explain ByteTrack IDs are temporary |
| 4 | Hold up the bottle | `plastic_bottle #ID conf` appears. Move it with you → label shows `carried by #ID` |
| 5 | Put a bottle on the floor first, walk past it | **No incident** – it was never carried (proximity ≠ guilt) |
| 6 | Staged drop: carry the bottle across the view for ~2 s, drop it into the lower part of the frame, step away | Label turns `POSSIBLE_DROP` → `AWAITING_VERIFICATION`; "Incidents this session" increments; recent events list shows reasons |
| 7 | **Stop camera** | Camera light turns off |
| 8 | Incident Review | Screenshot + ~6 s clip (3 s before / 3 s after), reasons, possible person track ID with association confidence, raw metadata |
| 9 | Confirm → pick fictional student → notes | Explain only an admin can confirm; DB trigger blocks fines on unconfirmed incidents |
| 10 | Generate DEMO receipt (set amount) | Show PDF with DEMONSTRATION ONLY watermark |
| 11 | Overview tab | KPIs and charts update |
| 12 | Student Portal → sign in with the chosen USN, PIN `1234` | Profile, fine history, incident details, download PDF, **Simulate payment** |
| 13 | Sign in as another student | Sees no fines (per-student access control) |
| 14 | Reject path (optional) | Reject an `UNCERTAIN` incident – no fine possible |

If the drop is not detected: the bottle probably fell out of view (→ `UNCERTAIN` after 2 s) or was
too small/blurred. Drop it closer to the camera, slower, onto a visible surface, and keep it in view
for ≥ 1 s after landing. Thresholds are in `config.py → ActionConfig` (e.g. lower `ground_zone`).

## Viva talking points
* **Why not just proximity?** It would blame bystanders. We require co-movement first, store the
  association as *possible* with a confidence, and require human verification.
* **Why UNCERTAIN?** Lost objects, items placed on tables, and crowded scenes are ambiguous; the
  system says so instead of guessing.
* **What can't it detect?** Paper/wrappers/packaging need the custom model (training pipeline in
  `training/`, annotation guide included).
* **Metrics**: mAP/precision/recall for the detector (`training/evaluate.py`); event-level
  precision/recall/F1 and false alarms/hour for actions using staged videos (`cli.py video`).
