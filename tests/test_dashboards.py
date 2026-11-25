"""Headless UI tests of the Streamlit dashboards (streamlit.testing AppTest).

They run against an isolated temporary database. The incident used here is a
TEST FIXTURE inserted directly - it is not, and is never shown as, an AI detection.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import database.db as dbmod
from database.seed import seed
from services import incidents as svc

PAGES = Path(__file__).resolve().parents[1] / "dashboards"


@pytest.fixture()
def tmpdb(tmp_path, monkeypatch):
    p = tmp_path / "ui.db"
    monkeypatch.setattr(dbmod, "DB_PATH", p)
    monkeypatch.setattr("services.receipts.RECEIPTS_DIR", tmp_path / "receipts")
    seed()
    return p


def _fixture_incident():
    ev = {"state": "SUSPECTED_LITTERING", "object_label": "plastic_bottle", "object_track_id": 3,
          "person_track_id": 1, "association_conf": 0.7, "confidence": 0.55,
          "reasons": ["TEST FIXTURE - not an AI detection"]}
    return svc.create_incident(ev, {})


def _no_errors(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_home_renders(tmpdb):
    at = AppTest.from_file(str(PAGES / "home.py"), default_timeout=30).run()
    _no_errors(at)
    assert "Smart Campus AI" in at.markdown[0].value or any("Smart Campus AI" in m.value for m in at.markdown)


def test_admin_requires_login(tmpdb):
    at = AppTest.from_file(str(PAGES / "admin.py"), default_timeout=30).run()
    _no_errors(at)
    assert len(at.tabs) == 0            # nothing but the sign-in form is rendered
    at.text_input[1].input("wrong").run()
    at.button[0].click().run()
    assert any("Invalid credentials" in e.value for e in at.error)
    at = AppTest.from_file(str(PAGES / "admin.py"), default_timeout=30).run()
    at.text_input[1].input("admin@123")
    at.button[0].click().run()
    _no_errors(at)
    assert len(at.tabs) == 5


def test_admin_confirm_and_fine_then_student_sees_it(tmpdb):
    iid = _fixture_incident()
    at = AppTest.from_file(str(PAGES / "admin.py"), default_timeout=60)
    at.session_state["admin_user"] = "admin"
    at.run()
    _no_errors(at)
    # review form: pick first fictional student, confirm
    confirm = next(b for b in at.button if "Confirm" in b.label)
    confirm.click().run()
    _no_errors(at)
    assert svc.get_incident(iid)["status"] == "CONFIRMED"
    at.selectbox[0].set_value("CONFIRMED").run()
    gen = next(b for b in at.button if "Generate DEMO receipt" in b.label)
    gen.click().run()
    _no_errors(at)
    fines = svc.list_fines()
    assert len(fines) == 1 and fines[0]["incident_id"] == iid
    student_id = fines[0]["student_id"]

    st_app = AppTest.from_file(str(PAGES / "student.py"), default_timeout=60)
    me = next(s for s in svc.list_students() if s["id"] == student_id)
    st_app.session_state["student"] = me
    st_app.run()
    _no_errors(st_app)
    assert any(fines[0]["receipt_no"] in str(df.value.values) for df in st_app.dataframe)
    pay = next(b for b in st_app.button if "Simulate payment" in b.label)
    pay.click().run()
    _no_errors(st_app)
    assert svc.get_fine(fines[0]["receipt_no"])["status"] == "PAID_SIMULATED"


def test_other_student_sees_no_fines(tmpdb):
    iid = _fixture_incident()
    s1, s2 = svc.list_students()[:2]
    svc.confirm_incident(iid, s1["id"], "admin")
    svc.issue_fine(iid, 200, "admin")
    at = AppTest.from_file(str(PAGES / "student.py"), default_timeout=30)
    at.session_state["student"] = s2
    at.run()
    _no_errors(at)
    assert len(at.dataframe) == 0
