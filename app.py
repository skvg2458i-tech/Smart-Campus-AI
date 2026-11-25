"""Smart Campus AI - Streamlit entry point.

Run:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from config import APP_NAME, ensure_dirs
from database.seed import seed

st.set_page_config(page_title=APP_NAME, page_icon="♻️", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner=False)
def _bootstrap() -> dict:
    ensure_dirs()
    return seed()  # idempotent: creates schema + fictional demo records once


_bootstrap()

from dashboards import theme  # noqa: E402

theme.apply()

pages = {
    "Overview": [st.Page("dashboards/home.py", title="Home", icon=":material/home:", default=True)],
    "Administrator": [st.Page("dashboards/admin.py", title="Admin Console", icon=":material/admin_panel_settings:")],
    "Students": [st.Page("dashboards/student.py", title="Student Portal", icon=":material/school:")],
}
with st.sidebar:
    st.markdown(f"### ♻️ {APP_NAME}")
    st.caption("BNMIT · AIML project · local demo")
nav = st.navigation(pages)
nav.run()
