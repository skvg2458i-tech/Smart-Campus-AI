"""Seed the database with a demo admin and FICTIONAL students.

All names/USNs below are invented for demonstration. USNs use a fake "1XX"
college code so they cannot collide with real BNMIT USNs.

Run:  python -m database.seed
"""
from __future__ import annotations

from config import DEMO_ADMIN_PASSWORD, DEMO_ADMIN_USERNAME, DEMO_STUDENT_PIN, settings
from database.db import connect, hash_secret, init_db, now_iso

FICTIONAL_STUDENTS = [
    # usn,          name,              department,                                   sem
    ("1XX22AI001", "Aarav Demo",       "Artificial Intelligence & Machine Learning", 5),
    ("1XX22AI002", "Diya Sample",      "Artificial Intelligence & Machine Learning", 5),
    ("1XX22CS003", "Rohan Testkumar",  "Computer Science & Engineering",             5),
    ("1XX22EC004", "Meera Placeholder", "Electronics & Communication Engineering",   5),
    ("1XX23IS005", "Kabir Fictional",  "Information Science & Engineering",          3),
    ("1XX23ME006", "Ananya Mockrao",   "Mechanical Engineering",                     3),
]


def seed(db_path=None) -> dict:
    init_db(db_path)
    added = 0
    with connect(db_path) as conn:
        if not conn.execute("SELECT 1 FROM admins WHERE username=?", (DEMO_ADMIN_USERNAME,)).fetchone():
            conn.execute(
                "INSERT INTO admins(username, password_hash, created_at) VALUES (?,?,?)",
                (DEMO_ADMIN_USERNAME, hash_secret(DEMO_ADMIN_PASSWORD), now_iso()),
            )
        for usn, name, dept, sem in FICTIONAL_STUDENTS:
            if conn.execute("SELECT 1 FROM students WHERE usn=?", (usn,)).fetchone():
                continue
            email = f"{usn.lower()}@demo.invalid"
            conn.execute(
                "INSERT INTO students(usn,name,department,semester,email,pin_hash,created_at) VALUES (?,?,?,?,?,?,?)",
                (usn, name, dept, sem, email, hash_secret(DEMO_STUDENT_PIN), now_iso()),
            )
            added += 1
        conn.execute(
            "INSERT OR IGNORE INTO app_settings(key,value) VALUES ('fine_amount', ?)",
            (str(settings.fine.default_amount),),
        )
        total = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    return {"students_added": added, "students_total": total}


if __name__ == "__main__":
    print(seed())
