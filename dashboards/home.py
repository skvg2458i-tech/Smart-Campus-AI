import streamlit as st

from dashboards import theme
from services import incidents as svc

theme.hero()

s = svc.stats()
c = st.columns(4)
theme.kpi(c[0], "Frames analysed", f"{s['frames']:,}", f"{s['sessions']} camera session(s)", True)
theme.kpi(c[1], "Suspected incidents", s["incidents"], f"{s['awaiting']} awaiting review", True)
theme.kpi(c[2], "Confirmed", s["confirmed"], f"{s['rejected']} rejected", True)
theme.kpi(c[3], "Demo fines issued", s["fines"], f"{s['fines_paid']} paid (simulated)", True)

st.write("")
theme.section("How it works")
cols = st.columns(4)
steps = [
    ("1 · Detect", "Pretrained YOLO finds people and supported waste objects (bottles, cups) in live webcam frames."),
    ("2 · Track", "ByteTrack gives each box a temporary, anonymous ID so movement can be followed across frames."),
    ("3 · Analyse", "A temporal rule baseline flags objects that were carried, then dropped and left on the ground."),
    ("4 · Verify", "Evidence is saved locally. Only an administrator can confirm an incident and issue a demo fine."),
]
for col, (h, p) in zip(cols, steps):
    col.markdown(f'<div class="scai-feature"><h4>{h}</h4><p>{p}</p></div>', unsafe_allow_html=True)

st.write("")
left, right = st.columns(2)
with left:
    theme.section("Choose a portal")
    a, b = st.columns(2)
    if a.button("Administrator console", type="primary", width="stretch"):
        st.switch_page("dashboards/admin.py")
    if b.button("Student portal", width="stretch"):
        st.switch_page("dashboards/student.py")
with right:
    theme.section("Ethics & safety")
    theme.note(
        "No facial recognition · person IDs are temporary tracker numbers, not identities · "
        "proximity never proves who littered · every incident needs administrator approval · "
        "all students are fictional · receipts are <b>DEMONSTRATION ONLY</b> · evidence stays on this laptop."
    )
