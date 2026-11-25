import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config import settings
from dashboards import theme
from database.db import set_setting
from services import incidents as svc
from services.pipeline import MonitorService

theme.hero("Administrator Console", "Live monitoring · incident verification · demonstration fines")

# ------------------------------------------------------------------ login
if not st.session_state.get("admin_user"):
    _, mid, _ = st.columns([1, 1.3, 1])
    with mid, st.form("admin_login"):
        st.subheader("Administrator sign-in")
        u = st.text_input("Username", value="admin")
        p = st.text_input("Password", type="password", help="Demo password is in README.md")
        if st.form_submit_button("Sign in", type="primary", width="stretch"):
            if svc.authenticate_admin(u, p):
                st.session_state["admin_user"] = u
                st.rerun()
            else:
                st.error("Invalid credentials")
    st.stop()

admin = st.session_state["admin_user"]
with st.sidebar:
    st.markdown(f"🔐 Signed in as **{admin}**")
    if st.button("Sign out", width="stretch"):
        st.session_state.pop("admin_user", None)
        st.rerun()


@st.cache_resource(show_spinner=False)
def monitor() -> MonitorService:
    return MonitorService()


tab_over, tab_live, tab_review, tab_fines, tab_settings = st.tabs(
    ["📊 Overview", "🎥 Live Monitoring", "🧾 Incident Review", "💳 Fines & Receipts", "⚙️ Settings"])

# --------------------------------------------------------------- overview
with tab_over:
    s = svc.stats()
    c = st.columns(5)
    theme.kpi(c[0], "Total detections", f"{s['person_detections'] + s['waste_detections']:,}",
              f"{s['person_detections']:,} person · {s['waste_detections']:,} waste (per frame)", True)
    theme.kpi(c[1], "Suspected incidents", s["incidents"], "raised by AI", True)
    theme.kpi(c[2], "Awaiting review", s["awaiting"], "need a decision", True)
    theme.kpi(c[3], "Confirmed", s["confirmed"], f"{s['rejected']} rejected", True)
    theme.kpi(c[4], "Demo fines", f"₹{s['fines_total']:,.0f}", f"{s['fines']} receipts", True)
    st.write("")
    incs = pd.DataFrame(svc.list_incidents())
    if incs.empty:
        theme.note("No incidents yet. Start <b>Live Monitoring</b> and stage a bottle drop to generate one.")
    else:
        l, r = st.columns(2)
        with l:
            fig = px.pie(incs, names="status", hole=.55, color="status", color_discrete_map=theme.CHART_COLORS,
                         title="Verification status")
            fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=320, legend_title_text="")
            st.plotly_chart(fig, width="stretch")
        with r:
            incs["date"] = pd.to_datetime(incs["created_at"]).dt.date
            daily = incs.groupby(["date", "detection_category"]).size().reset_index(name="count")
            fig = px.bar(daily, x="date", y="count", color="detection_category",
                         color_discrete_map=theme.CHART_COLORS, title="Incidents per day by AI category")
            fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=320, legend_title_text="")
            st.plotly_chart(fig, width="stretch")
        l, r = st.columns(2)
        with l:
            fig = px.histogram(incs, x="confidence", nbins=10, title="AI confidence distribution",
                               color_discrete_sequence=[theme.ACCENT])
            fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=300)
            st.plotly_chart(fig, width="stretch")
        with r:
            obj = incs.groupby("object_category").size().reset_index(name="count")
            fig = px.bar(obj, x="object_category", y="count", title="Object categories",
                         color_discrete_sequence=[theme.PRIMARY])
            fig.update_layout(margin=dict(t=50, b=10, l=10, r=10), height=300)
            st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------- live
with tab_live:
    mon = monitor()
    b1, b2, info = st.columns([1, 1, 3])
    # on_click callbacks fire exactly once per real click (never replayed on reruns)
    b1.button("▶ Start camera", key="cam_start", type="primary", width="stretch",
              disabled=mon.running or mon.loading, on_click=mon.start)
    b2.button("■ Stop camera", key="cam_stop", width="stretch",
              disabled=not (mon.running or mon.loading), on_click=mon.stop)
    info.caption("Frames are processed locally on this laptop. Nothing is uploaded. "
                 "Webcam index and thresholds are set in config.py.")

    @st.fragment(run_every=0.3)
    def live_view():
        snap = mon.snapshot()
        feed, side = st.columns([3, 1.2])
        with feed:
            if snap["error"]:
                st.error(f"Camera/pipeline error: {snap['error']}")
            if snap["loading"]:
                st.info("Loading YOLO model and opening the webcam…")
            if mon.latest_jpeg and (snap["running"]):
                st.image(mon.latest_jpeg, width="stretch")
            elif not snap["loading"]:
                st.markdown('<div class="scai-card" style="height:360px;display:flex;align-items:center;'
                            'justify-content:center;color:#667085">Camera stopped</div>', unsafe_allow_html=True)
        with side:
            state = "🟢 LIVE" if snap["running"] else ("🟡 Starting" if snap["loading"] else "⚪ Stopped")
            st.markdown(f"**Status:** {state}")
            st.metric("FPS", snap["fps"])
            cnt = snap["counters"]
            st.metric("Frames", cnt.get("frames", 0))
            st.metric("Person tracks", snap["unique_persons"])
            st.metric("Waste tracks", snap["unique_waste"])
            st.metric("Incidents this session", len(snap["incidents"]))
            st.caption(f"Detection status: {snap['status']}")
            if snap["model"]:
                m = snap["model"]
                st.caption(f"Model: {Path(m['person_model']).name} · waste: {m['waste_source']} "
                           f"({', '.join(m['waste_classes'])}) · tracker: {m['tracker']}")
        if snap["events"]:
            with st.expander(f"Recent action events ({len(snap['events'])})", expanded=False):
                for e in reversed(snap["events"]):
                    st.markdown(f"{theme.badge(e['state'])} `{e['event_time']}` {e['object_label']} "
                                f"#{e['object_track_id']} · possible person #{e['person_track_id']} "
                                f"(assoc {e['association_conf']:.2f}) · conf {e['confidence']:.2f}",
                                unsafe_allow_html=True)

    live_view()
    theme.note("<b>Staging a test:</b> hold a plastic bottle and walk across the view for 1–2 s, drop it so it lands "
               "in the lower part of the frame (below the grey <i>ground zone</i> line), and walk away. "
               "Walking past a bottle already on the floor should <b>not</b> create an incident.")

# ----------------------------------------------------------------- review
with tab_review:
    f1, f2 = st.columns([1, 3])
    flt = f1.selectbox("Filter", ["AWAITING_VERIFICATION", "CONFIRMED", "REJECTED", "All"])
    incs = svc.list_incidents(None if flt == "All" else flt)
    if not incs:
        theme.note("No incidents in this view.")
    else:
        df = pd.DataFrame(incs)[["id", "created_at", "detection_category", "object_category",
                                 "person_track_id", "association_conf", "confidence", "status", "student_usn"]]
        f2.dataframe(df, hide_index=True, width="stretch", height=200)
        sel = st.selectbox("Open incident", [i["id"] for i in incs],
                           format_func=lambda i: next(f"{x['id']} · {x['detection_category']} · {x['status']}"
                                                      for x in incs if x["id"] == i))
        inc = svc.get_incident(sel)
        left, right = st.columns([1.5, 1])
        with left:
            theme.section("Evidence")
            if svc.evidence_exists(inc["screenshot_path"]):
                st.image(inc["screenshot_path"], caption="Screenshot at event time", width="stretch")
            else:
                st.warning("Screenshot missing")
            if svc.evidence_exists(inc["clip_path"]):
                st.video(inc["clip_path"])
                st.caption("Clip: ~3 s before and after the event")
            else:
                st.info("Clip is still being written (needs ~3 s of post-event frames) or the camera was stopped.")
        with right:
            theme.section("Details")
            st.markdown(f"**ID:** `{inc['id']}`  \n**Status:** {theme.badge(inc['status'])}  \n"
                        f"**AI category:** {theme.badge(inc['detection_category'])}  \n"
                        f"**Event time:** {inc['event_time']}  \n**Object:** {inc['object_category']} "
                        f"(track #{inc['object_track_id']})  \n"
                        f"**Possible person track:** #{inc['person_track_id']} "
                        f"(association {inc['association_conf'] or 0:.2f})  \n"
                        f"**Confidence:** {inc['confidence']:.2f}", unsafe_allow_html=True)
            with st.expander("Why the AI flagged this", expanded=True):
                for r in json.loads(inc["reasons"] or "[]"):
                    st.markdown(f"- {r}")
            if inc["metadata_path"] and Path(inc["metadata_path"]).exists():
                with st.expander("Raw detection metadata"):
                    st.json(json.loads(Path(inc["metadata_path"]).read_text()), expanded=False)
            theme.note("A track ID is <b>not</b> an identity. Confirm only if the clip clearly shows the "
                       "litter-dropping action. The student is chosen by the administrator from fictional records.")
            if inc["status"] == "AWAITING_VERIFICATION":
                students = svc.list_students()
                with st.form(f"review_{sel}"):
                    stu = st.selectbox("Assign fictional student", students,
                                       format_func=lambda s: f"{s['usn']} · {s['name']} · {s['department']}")
                    notes = st.text_area("Review notes", placeholder="What you saw in the evidence")
                    c1, c2 = st.columns(2)
                    ok = c1.form_submit_button("✔ Confirm", type="primary", width="stretch")
                    rej = c2.form_submit_button("✖ Reject", width="stretch")
                if ok:
                    svc.confirm_incident(sel, stu["id"], admin, notes)
                    st.success("Incident confirmed. Issue the demo fine below.")
                    st.rerun()
                if rej:
                    svc.reject_incident(sel, admin, notes)
                    st.warning("Incident rejected.")
                    st.rerun()
            else:
                st.markdown(f"Reviewed by **{inc['reviewed_by']}** at {inc['reviewed_at']}  \n"
                            f"Notes: {inc['review_notes'] or '-'}")
                if inc["status"] == "CONFIRMED":
                    st.markdown(f"Assigned student: **{inc['student_usn']} · {inc['student_name']}**")
                    fine = next((f for f in svc.list_fines() if f["incident_id"] == sel), None)
                    if fine:
                        st.success(f"Receipt {fine['receipt_no']} issued · ₹{fine['amount']:.0f} · "
                                   f"{fine['status']}")
                    else:
                        with st.form(f"fine_{sel}"):
                            amt = st.number_input("Demo fine amount (INR)", min_value=settings.fine.min_amount,
                                                  max_value=settings.fine.max_amount,
                                                  value=svc.current_fine_amount(), step=50.0)
                            if st.form_submit_button("Generate DEMO receipt", type="primary", width="stretch"):
                                f = svc.issue_fine(sel, amt, admin)
                                st.success(f"Receipt {f['receipt_no']} generated")
                                st.rerun()

# ------------------------------------------------------------------ fines
with tab_fines:
    fines = svc.list_fines()
    if not fines:
        theme.note("No demonstration fines issued yet.")
    else:
        df = pd.DataFrame(fines)[["receipt_no", "incident_id", "usn", "student_name", "department", "amount",
                                  "status", "issued_at", "paid_at"]]
        st.dataframe(df, hide_index=True, width="stretch")
        rn = st.selectbox("Receipt", [f["receipt_no"] for f in fines])
        f = next(x for x in fines if x["receipt_no"] == rn)
        if f["receipt_path"] and Path(f["receipt_path"]).exists():
            st.download_button("⬇ Download PDF receipt", Path(f["receipt_path"]).read_bytes(),
                               file_name=f"{rn}.pdf", mime="application/pdf")

# --------------------------------------------------------------- settings
with tab_settings:
    theme.section("Demonstration fine")
    with st.form("fine_cfg"):
        amt = st.number_input("Default fine amount (INR)", min_value=settings.fine.min_amount,
                              max_value=settings.fine.max_amount, value=svc.current_fine_amount(), step=50.0)
        if st.form_submit_button("Save", type="primary"):
            set_setting("fine_amount", str(amt), actor=admin)
            st.success("Saved")
    theme.section("AI thresholds (read-only - edit config.py)")
    st.json({"model": {k: str(v) for k, v in vars(settings.model).items()},
             "action_analysis": vars(settings.action), "evidence": vars(settings.evidence)}, expanded=False)
    theme.section("Students (fictional)")
    st.dataframe(pd.DataFrame(svc.list_students()), hide_index=True, width="stretch")
