import json
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboards import theme
from services import incidents as svc

theme.hero("Student Portal", "View demonstration fines linked to your fictional student record")

if not st.session_state.get("student"):
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid, st.form("student_login"):
        st.subheader("Demo student sign-in")
        students = svc.list_students()
        usn = st.selectbox("USN (fictional)", [s["usn"] for s in students],
                           format_func=lambda u: f"{u} · {next(s['name'] for s in students if s['usn'] == u)}")
        pin = st.text_input("PIN", type="password", help="Demo PIN for all fictional students is in README.md")
        if st.form_submit_button("Sign in", type="primary", width="stretch"):
            s = svc.authenticate_student(usn, pin)
            if s:
                st.session_state["student"] = s
                st.rerun()
            else:
                st.error("Invalid USN or PIN")
    st.stop()

me = st.session_state["student"]
with st.sidebar:
    st.markdown(f"🎓 Signed in as **{me['usn']}**")
    if st.button("Sign out", width="stretch"):
        st.session_state.pop("student", None)
        st.rerun()

fines = svc.list_fines(me["id"])  # only this student's records are ever queried
c = st.columns(4)
theme.kpi(c[0], "Student", me["name"], me["usn"], True)
theme.kpi(c[1], "Department", me["department"].split("&")[0].strip(), f"Semester {me['semester']}", True)
theme.kpi(c[2], "Demo fines", len(fines), f"₹{sum(f['amount'] for f in fines):,.0f} total", True)
unpaid = [f for f in fines if f["status"] == "UNPAID"]
theme.kpi(c[3], "Outstanding (demo)", f"₹{sum(f['amount'] for f in unpaid):,.0f}", f"{len(unpaid)} unpaid", True)
st.write("")

tab_hist, tab_profile = st.tabs(["🧾 Fine history", "👤 Profile"])
with tab_profile:
    st.markdown(f"""
| | |
|---|---|
| **Name** | {me['name']} |
| **USN** | {me['usn']} |
| **Department** | {me['department']} |
| **Semester** | {me['semester']} |
| **Email** | {me['email']} |
""")
    theme.note("This is a <b>fictional</b> student record created for a college demonstration.")

with tab_hist:
    if not fines:
        theme.note("No fines on record. 🎉")
    else:
        df = pd.DataFrame(fines)[["receipt_no", "incident_id", "event_time", "object_category", "amount", "status"]]
        st.dataframe(df, hide_index=True, width="stretch")
        rn = st.selectbox("Open receipt", [f["receipt_no"] for f in fines])
        f = svc.student_fine_for(me["id"], rn)  # access check
        full = next(x for x in fines if x["receipt_no"] == rn)
        l, r = st.columns([1.2, 1])
        with l:
            theme.section("Incident details")
            st.markdown(f"**Incident:** `{f['incident_id']}`  \n**Event time:** {f['event_time']}  \n"
                        f"**Object:** {f['object_category']}  \n**AI category:** {f['detection_category']} "
                        f"(confidence {f['confidence']:.2f})  \n**Verified by:** {f['reviewed_by']} at "
                        f"{f['reviewed_at']}  \n**Admin notes:** {full.get('review_notes') or '-'}")
            if svc.evidence_exists(full.get("screenshot_path")):
                st.image(full["screenshot_path"], caption="Evidence snapshot", width="stretch")
        with r:
            theme.section("Receipt")
            st.markdown(f"**Receipt:** {f['receipt_no']}  \n**Amount:** {f['currency']} {f['amount']:,.2f}  \n"
                        f"**Status:** {theme.badge(f['status'])}", unsafe_allow_html=True)
            p = Path(f["receipt_path"]) if f["receipt_path"] else None
            if p and p.exists():
                st.download_button("⬇ Download PDF receipt", p.read_bytes(), file_name=f"{rn}.pdf",
                                   mime="application/pdf", width="stretch")
            if f["status"] == "UNPAID":
                st.caption("Payment is simulated - no money is collected.")
                if st.button("Simulate payment", type="primary", width="stretch"):
                    svc.simulate_payment(me["id"], rn)
                    st.success("Simulated payment recorded (no real transaction).")
                    st.rerun()
            else:
                st.success(f"Paid (simulated) at {f['paid_at']}")
            theme.note("<b>DEMONSTRATION ONLY</b> - not a real fine.")
