"""SQLite persistence layer.

Relationships
-------------
students 1---* incidents (assigned_student_id, set only on admin confirmation)
incidents 1---1 fines (fines.incident_id UNIQUE)
students 1---* fines
monitor_sessions: aggregated detection counters per camera session

Integrity rules enforced *inside SQLite* (not just the UI):
* a fine can only be inserted for an incident whose status is CONFIRMED
* a fine's student must be the student assigned to that incident
* a confirmed incident must have an assigned student
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from config import DB_PATH, ensure_dirs

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS admins (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS students (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    usn         TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    department  TEXT NOT NULL,
    semester    INTEGER NOT NULL,
    email       TEXT NOT NULL,
    pin_hash    TEXT NOT NULL,
    is_fictional INTEGER NOT NULL DEFAULT 1 CHECK (is_fictional = 1),
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS incidents (
    id                   TEXT PRIMARY KEY,
    created_at           TEXT NOT NULL,
    event_time           TEXT NOT NULL,
    detection_category   TEXT NOT NULL,          -- SUSPECTED_LITTERING | UNCERTAIN
    object_category      TEXT NOT NULL,
    object_track_id      INTEGER,
    person_track_id      INTEGER,                -- temporary ByteTrack ID: a *possible* association only
    association_conf     REAL,
    confidence           REAL NOT NULL,
    reasons              TEXT,                   -- JSON list of analysis reasons
    screenshot_path      TEXT,
    clip_path            TEXT,
    metadata_path        TEXT,
    status               TEXT NOT NULL DEFAULT 'AWAITING_VERIFICATION'
                          CHECK (status IN ('AWAITING_VERIFICATION','CONFIRMED','REJECTED')),
    assigned_student_id  INTEGER REFERENCES students(id) ON DELETE RESTRICT,
    reviewed_by          TEXT,
    reviewed_at          TEXT,
    review_notes         TEXT,
    session_id           INTEGER REFERENCES monitor_sessions(id),
    CHECK (status != 'CONFIRMED' OR assigned_student_id IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS fines (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_no   TEXT NOT NULL UNIQUE,
    incident_id  TEXT NOT NULL UNIQUE REFERENCES incidents(id) ON DELETE RESTRICT,
    student_id   INTEGER NOT NULL REFERENCES students(id) ON DELETE RESTRICT,
    amount       REAL NOT NULL CHECK (amount > 0),
    currency     TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'UNPAID' CHECK (status IN ('UNPAID','PAID_SIMULATED')),
    issued_by    TEXT NOT NULL,
    issued_at    TEXT NOT NULL,
    paid_at      TEXT,
    receipt_path TEXT
);

CREATE TRIGGER IF NOT EXISTS fines_require_confirmed
BEFORE INSERT ON fines
FOR EACH ROW
WHEN (SELECT status FROM incidents WHERE id = NEW.incident_id) IS NOT 'CONFIRMED'
  OR (SELECT assigned_student_id FROM incidents WHERE id = NEW.incident_id) IS NOT NEW.student_id
BEGIN
    SELECT RAISE(ABORT, 'Fine requires an administrator-CONFIRMED incident assigned to the same student');
END;

CREATE TRIGGER IF NOT EXISTS incidents_lock_after_fine
BEFORE UPDATE OF status, assigned_student_id ON incidents
FOR EACH ROW
WHEN EXISTS (SELECT 1 FROM fines WHERE incident_id = OLD.id)
BEGIN
    SELECT RAISE(ABORT, 'Incident already has a fine; it can no longer be changed');
END;

CREATE TABLE IF NOT EXISTS monitor_sessions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at        TEXT NOT NULL,
    ended_at          TEXT,
    frames            INTEGER NOT NULL DEFAULT 0,
    person_detections INTEGER NOT NULL DEFAULT 0,
    waste_detections  INTEGER NOT NULL DEFAULT 0,
    unique_persons    INTEGER NOT NULL DEFAULT 0,
    unique_waste      INTEGER NOT NULL DEFAULT 0,
    incidents         INTEGER NOT NULL DEFAULT 0,
    avg_fps           REAL
);

CREATE TABLE IF NOT EXISTS app_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    at         TEXT NOT NULL,
    actor      TEXT NOT NULL,
    action     TEXT NOT NULL,
    target     TEXT,
    details    TEXT
);

CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_fines_student ON fines(student_id);
"""


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- passwords
def hash_secret(secret: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", secret.encode(), salt, 120_000)
    return f"{salt.hex()}${digest.hex()}"


def verify_secret(secret: str, stored: str) -> bool:
    try:
        salt_hex, digest_hex = stored.split("$", 1)
    except ValueError:
        return False
    test = hashlib.pbkdf2_hmac("sha256", secret.encode(), bytes.fromhex(salt_hex), 120_000)
    return hmac.compare_digest(test.hex(), digest_hex)


# ---------------------------------------------------------------- connection
@contextmanager
def connect(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str | None = None) -> Path:
    ensure_dirs()
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
    return Path(db_path or DB_PATH)


def rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def audit(conn: sqlite3.Connection, actor: str, action: str, target: str | None = None, details: dict | None = None) -> None:
    conn.execute(
        "INSERT INTO audit_log(at, actor, action, target, details) VALUES (?,?,?,?,?)",
        (now_iso(), actor, action, target, json.dumps(details or {})),
    )


def new_incident_id() -> str:
    return f"INC-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6].upper()}"


# ---------------------------------------------------------------- settings
def get_setting(key: str, default: str | None = None, db_path=None) -> str | None:
    with connect(db_path) as conn:
        r = conn.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
    return r["value"] if r else default


def set_setting(key: str, value: str, actor: str = "system", db_path=None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO app_settings(key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        audit(conn, actor, "SET_SETTING", key, {"value": value})
