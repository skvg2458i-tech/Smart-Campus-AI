"""Incident, fine and student business logic (all access-controlled operations live here)."""
from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path

from config import settings
from database.db import audit, connect, get_setting, new_incident_id, now_iso, rows, verify_secret
from services.logger import get_logger

log = get_logger("incidents")


class AccessDenied(PermissionError):
    pass


# ------------------------------------------------------------------ auth
def authenticate_admin(username: str, password: str, db_path=None) -> bool:
    with connect(db_path) as conn:
        r = conn.execute("SELECT password_hash FROM admins WHERE username=?", (username,)).fetchone()
        ok = bool(r) and verify_secret(password, r["password_hash"])
        audit(conn, username or "?", "ADMIN_LOGIN_OK" if ok else "ADMIN_LOGIN_FAIL")
    return ok


def authenticate_student(usn: str, pin: str, db_path=None) -> dict | None:
    usn = (usn or "").strip().upper()
    with connect(db_path) as conn:
        r = conn.execute("SELECT * FROM students WHERE usn=?", (usn,)).fetchone()
        ok = bool(r) and verify_secret(pin, r["pin_hash"])
        audit(conn, usn or "?", "STUDENT_LOGIN_OK" if ok else "STUDENT_LOGIN_FAIL")
    if not ok:
        return None
    s = dict(r)
    s.pop("pin_hash", None)
    return s


# ------------------------------------------------------------- incidents
def create_incident(event: dict, evidence: dict, session_id: int | None = None, db_path=None,
                    incident_id: str | None = None) -> str:
    iid = incident_id or new_incident_id()
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO incidents(id, created_at, event_time, detection_category, object_category,
                   object_track_id, person_track_id, association_conf, confidence, reasons,
                   screenshot_path, clip_path, metadata_path, session_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (iid, now_iso(), event.get("event_time", now_iso()), event["state"], event["object_label"],
             event.get("object_track_id"), event.get("person_track_id"), event.get("association_conf"),
             event["confidence"], json.dumps(event.get("reasons", [])), evidence.get("screenshot_path"),
             None, evidence.get("metadata_path"), session_id),
        )
        if session_id:
            conn.execute("UPDATE monitor_sessions SET incidents = incidents + 1 WHERE id=?", (session_id,))
        audit(conn, "ai-pipeline", "INCIDENT_CREATED", iid, {"state": event["state"]})
    log.info("Incident %s created (%s, conf %.2f)", iid, event["state"], event["confidence"])
    return iid


def set_clip_path(incident_id: str, clip_path: str, db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE incidents SET clip_path=? WHERE id=?", (clip_path, incident_id))


def list_incidents(status: str | None = None, db_path=None) -> list[dict]:
    sql = """SELECT i.*, s.usn AS student_usn, s.name AS student_name, f.receipt_no, f.amount AS fine_amount,
                    f.status AS fine_status
             FROM incidents i
             LEFT JOIN students s ON s.id = i.assigned_student_id
             LEFT JOIN fines f ON f.incident_id = i.id"""
    params: tuple = ()
    if status:
        sql += " WHERE i.status=?"
        params = (status,)
    sql += " ORDER BY i.created_at DESC"
    with connect(db_path) as conn:
        return rows(conn, sql, params)


def get_incident(incident_id: str, db_path=None) -> dict | None:
    with connect(db_path) as conn:
        r = rows(conn, """SELECT i.*, s.usn AS student_usn, s.name AS student_name
                          FROM incidents i LEFT JOIN students s ON s.id=i.assigned_student_id
                          WHERE i.id=?""", (incident_id,))
    return r[0] if r else None


def confirm_incident(incident_id: str, student_id: int, admin: str, notes: str = "", db_path=None) -> None:
    with connect(db_path) as conn:
        inc = conn.execute("SELECT status FROM incidents WHERE id=?", (incident_id,)).fetchone()
        if not inc:
            raise ValueError("Incident not found")
        if inc["status"] != "AWAITING_VERIFICATION":
            raise ValueError(f"Incident is already {inc['status']}")
        if not conn.execute("SELECT 1 FROM students WHERE id=?", (student_id,)).fetchone():
            raise ValueError("Student not found")
        conn.execute(
            """UPDATE incidents SET status='CONFIRMED', assigned_student_id=?, reviewed_by=?, reviewed_at=?,
               review_notes=? WHERE id=?""",
            (student_id, admin, now_iso(), notes, incident_id),
        )
        audit(conn, admin, "INCIDENT_CONFIRMED", incident_id, {"student_id": student_id, "notes": notes})


def reject_incident(incident_id: str, admin: str, notes: str = "", db_path=None) -> None:
    with connect(db_path) as conn:
        inc = conn.execute("SELECT status FROM incidents WHERE id=?", (incident_id,)).fetchone()
        if not inc:
            raise ValueError("Incident not found")
        if inc["status"] != "AWAITING_VERIFICATION":
            raise ValueError(f"Incident is already {inc['status']}")
        conn.execute(
            "UPDATE incidents SET status='REJECTED', reviewed_by=?, reviewed_at=?, review_notes=? WHERE id=?",
            (admin, now_iso(), notes, incident_id),
        )
        audit(conn, admin, "INCIDENT_REJECTED", incident_id, {"notes": notes})


# ----------------------------------------------------------------- fines
def current_fine_amount(db_path=None) -> float:
    return float(get_setting("fine_amount", str(settings.fine.default_amount), db_path))


def issue_fine(incident_id: str, amount: float, admin: str, db_path=None) -> dict:
    """Create a DEMONSTRATION fine + PDF receipt. Only CONFIRMED incidents qualify (also enforced by a DB trigger)."""
    from services.receipts import generate_receipt

    fc = settings.fine
    if not (fc.min_amount <= amount <= fc.max_amount):
        raise ValueError(f"Fine must be between {fc.min_amount:.0f} and {fc.max_amount:.0f} {fc.currency}")
    inc = get_incident(incident_id, db_path)
    if not inc:
        raise ValueError("Incident not found")
    if inc["status"] != "CONFIRMED":
        raise AccessDenied("Only administrator-confirmed incidents can generate fines")
    receipt_no = f"DEMO-RCPT-{datetime.now():%Y%m%d}-{secrets.token_hex(3).upper()}"
    with connect(db_path) as conn:
        conn.execute(
            """INSERT INTO fines(receipt_no, incident_id, student_id, amount, currency, issued_by, issued_at)
               VALUES (?,?,?,?,?,?,?)""",
            (receipt_no, incident_id, inc["assigned_student_id"], amount, fc.currency, admin, now_iso()),
        )
        audit(conn, admin, "FINE_ISSUED", incident_id, {"receipt_no": receipt_no, "amount": amount})
    fine = get_fine(receipt_no, db_path)
    path = generate_receipt(fine)
    with connect(db_path) as conn:
        conn.execute("UPDATE fines SET receipt_path=? WHERE receipt_no=?", (str(path), receipt_no))
    fine["receipt_path"] = str(path)
    return fine


def get_fine(receipt_no: str, db_path=None) -> dict | None:
    with connect(db_path) as conn:
        r = rows(conn, """SELECT f.*, s.usn, s.name AS student_name, s.department, s.semester,
                                 i.event_time, i.object_category, i.detection_category, i.confidence,
                                 i.reviewed_by, i.reviewed_at
                          FROM fines f JOIN students s ON s.id=f.student_id JOIN incidents i ON i.id=f.incident_id
                          WHERE f.receipt_no=?""", (receipt_no,))
    return r[0] if r else None


def list_fines(student_id: int | None = None, db_path=None) -> list[dict]:
    sql = """SELECT f.*, s.usn, s.name AS student_name, s.department, i.event_time, i.object_category,
                    i.detection_category, i.screenshot_path, i.review_notes
             FROM fines f JOIN students s ON s.id=f.student_id JOIN incidents i ON i.id=f.incident_id"""
    params: tuple = ()
    if student_id is not None:
        sql += " WHERE f.student_id=?"
        params = (student_id,)
    sql += " ORDER BY f.issued_at DESC"
    with connect(db_path) as conn:
        return rows(conn, sql, params)


def student_fine_for(student_id: int, receipt_no: str, db_path=None) -> dict:
    """Access control: a student may only open their own fine/receipt."""
    fine = get_fine(receipt_no, db_path)
    if not fine or fine["student_id"] != student_id:
        raise AccessDenied("This receipt does not belong to the logged-in student")
    return fine


def simulate_payment(student_id: int, receipt_no: str, db_path=None) -> None:
    fine = student_fine_for(student_id, receipt_no, db_path)
    if fine["status"] == "PAID_SIMULATED":
        return
    with connect(db_path) as conn:
        conn.execute("UPDATE fines SET status='PAID_SIMULATED', paid_at=? WHERE receipt_no=?", (now_iso(), receipt_no))
        audit(conn, fine["usn"], "PAYMENT_SIMULATED", receipt_no)
    from services.receipts import generate_receipt
    generate_receipt(get_fine(receipt_no, db_path))  # refresh PDF with paid status


def list_students(db_path=None) -> list[dict]:
    with connect(db_path) as conn:
        return rows(conn, "SELECT id, usn, name, department, semester, email FROM students ORDER BY usn")


# ----------------------------------------------------------------- stats
def stats(db_path=None) -> dict:
    with connect(db_path) as conn:
        q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "frames": q("SELECT COALESCE(SUM(frames),0) FROM monitor_sessions"),
            "person_detections": q("SELECT COALESCE(SUM(person_detections),0) FROM monitor_sessions"),
            "waste_detections": q("SELECT COALESCE(SUM(waste_detections),0) FROM monitor_sessions"),
            "unique_persons": q("SELECT COALESCE(SUM(unique_persons),0) FROM monitor_sessions"),
            "unique_waste": q("SELECT COALESCE(SUM(unique_waste),0) FROM monitor_sessions"),
            "sessions": q("SELECT COUNT(*) FROM monitor_sessions"),
            "incidents": q("SELECT COUNT(*) FROM incidents"),
            "awaiting": q("SELECT COUNT(*) FROM incidents WHERE status='AWAITING_VERIFICATION'"),
            "confirmed": q("SELECT COUNT(*) FROM incidents WHERE status='CONFIRMED'"),
            "rejected": q("SELECT COUNT(*) FROM incidents WHERE status='REJECTED'"),
            "fines": q("SELECT COUNT(*) FROM fines"),
            "fines_total": q("SELECT COALESCE(SUM(amount),0) FROM fines"),
            "fines_paid": q("SELECT COUNT(*) FROM fines WHERE status='PAID_SIMULATED'"),
        }


def evidence_exists(path: str | None) -> bool:
    return bool(path) and Path(path).exists()
