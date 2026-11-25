"""Database relationships, access control, receipts and evidence writing."""
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from database.db import connect, init_db
from database.seed import seed
from services import incidents as svc
from services.evidence import EvidenceRecorder, write_clip


@pytest.fixture()
def db(tmp_path, monkeypatch):
    p = tmp_path / "t.db"
    monkeypatch.setattr("services.receipts.RECEIPTS_DIR", tmp_path / "receipts")
    seed(p)
    return p


def _incident(db, tmp_path):
    ev = {"state": "SUSPECTED_LITTERING", "object_label": "plastic_bottle", "object_track_id": 7,
          "person_track_id": 1, "association_conf": 0.8, "confidence": 0.6, "reasons": ["test"]}
    return svc.create_incident(ev, {"screenshot_path": str(tmp_path / "s.jpg")}, db_path=db)


def test_seed_creates_fictional_students_and_admin(db):
    students = svc.list_students(db)
    assert len(students) == 6
    assert all(s["usn"].startswith("1XX") for s in students)
    assert svc.authenticate_admin("admin", "admin@123", db)
    assert not svc.authenticate_admin("admin", "wrong", db)
    assert svc.authenticate_student("1xx22ai001", "1234", db)["name"] == "Aarav Demo"
    assert svc.authenticate_student("1XX22AI001", "0000", db) is None


def test_fine_requires_confirmed_incident(db, tmp_path):
    iid = _incident(db, tmp_path)
    with pytest.raises(svc.AccessDenied):
        svc.issue_fine(iid, 200, "admin", db)
    # the DB trigger also blocks a direct insert that bypasses the service layer
    with pytest.raises(sqlite3.IntegrityError):
        with connect(db) as c:
            c.execute("INSERT INTO fines(receipt_no,incident_id,student_id,amount,currency,issued_by,issued_at)"
                      " VALUES ('X',?,1,100,'INR','admin','now')", (iid,))


def test_rejected_incident_cannot_be_fined_or_reconfirmed(db, tmp_path):
    iid = _incident(db, tmp_path)
    svc.reject_incident(iid, "admin", "false positive", db)
    with pytest.raises(ValueError):
        svc.confirm_incident(iid, 1, "admin", db_path=db)
    with pytest.raises(svc.AccessDenied):
        svc.issue_fine(iid, 200, "admin", db)


def test_full_review_flow_and_student_access_control(db, tmp_path):
    iid = _incident(db, tmp_path)
    s1, s2 = svc.list_students(db)[:2]
    svc.confirm_incident(iid, s1["id"], "admin", "verified on clip", db)
    fine = svc.issue_fine(iid, 250, "admin", db)
    pdf = Path(fine["receipt_path"])
    assert pdf.exists() and pdf.read_bytes()[:4] == b"%PDF"
    assert svc.list_fines(s1["id"], db)[0]["receipt_no"] == fine["receipt_no"]
    assert svc.list_fines(s2["id"], db) == []
    with pytest.raises(svc.AccessDenied):
        svc.student_fine_for(s2["id"], fine["receipt_no"], db)
    with pytest.raises(svc.AccessDenied):
        svc.simulate_payment(s2["id"], fine["receipt_no"], db)
    svc.simulate_payment(s1["id"], fine["receipt_no"], db)
    assert svc.get_fine(fine["receipt_no"], db)["status"] == "PAID_SIMULATED"
    # one fine per incident, and a fined incident is locked
    with pytest.raises(sqlite3.IntegrityError):
        svc.issue_fine(iid, 100, "admin", db)
    with pytest.raises(sqlite3.IntegrityError):
        with connect(db) as c:
            c.execute("UPDATE incidents SET assigned_student_id=? WHERE id=?", (s2["id"], iid))
    st = svc.stats(db)
    assert st["confirmed"] == 1 and st["fines"] == 1 and st["fines_paid"] == 1


def test_fine_amount_limits(db, tmp_path):
    iid = _incident(db, tmp_path)
    svc.confirm_incident(iid, 1, "admin", db_path=db)
    with pytest.raises(ValueError):
        svc.issue_fine(iid, 1_000_000, "admin", db)


def test_receipt_is_labelled_demonstration_only(db, tmp_path):
    iid = _incident(db, tmp_path)
    svc.confirm_incident(iid, 1, "admin", db_path=db)
    fine = svc.issue_fine(iid, 200, "admin", db)
    from pypdf import PdfReader
    text = PdfReader(fine["receipt_path"]).pages[0].extract_text()
    assert "DEMONSTRATION ONLY" in text and "1XX22AI001" in text


def test_evidence_clip_contains_pre_and_post_frames(tmp_path):
    rec = EvidenceRecorder(root=tmp_path)
    done = {}
    for i in range(40):                      # pre-event frames
        rec.push(np.full((240, 320, 3), i * 5, np.uint8), fps=10)
    out = rec.capture("INC-TEST", np.zeros((240, 320, 3), np.uint8), np.zeros((240, 320, 3), np.uint8),
                      {"k": 1}, fps=10, on_clip_done=lambda i, p: done.setdefault(i, p))
    for i in range(30):                      # post-event frames
        rec.push(np.full((240, 320, 3), 200, np.uint8), fps=10)
    rec.flush()
    assert Path(out["screenshot_path"]).exists() and Path(out["metadata_path"]).exists()
    assert "INC-TEST" in done
    cap = cv2.VideoCapture(done["INC-TEST"])
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n >= 50   # ~3 s pre + 3 s post at 10 fps


def test_write_clip_is_h264(tmp_path):
    p = write_clip([np.zeros((120, 160, 3), np.uint8)] * 10, tmp_path / "c.mp4", 10)
    data = p.read_bytes()
    assert b"avc1" in data[:4096] or b"avc1" in data   # H.264 -> plays in browsers
