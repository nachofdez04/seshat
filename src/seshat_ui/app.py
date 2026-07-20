from __future__ import annotations

import logging
import os

import streamlit as st

from seshat_ui.api_client import ApiClient
from seshat_ui.screens.admin import render as _render_admin
from seshat_ui.screens.graph import render as _render_kb
from seshat_ui.screens.jobs import render as _render_jobs
from seshat_ui.screens.manual_actions import render as _render_manual_actions
from seshat_ui.utils import job_label

logger = logging.getLogger(__name__)


API_BASE = os.environ.get("SESHAT_API_BASE", "http://localhost:8000")
MLFLOW_UI_BASE = os.environ.get("MLFLOW_UI_URL", "")

_HEALTH_CHIP: dict[str, str] = {
    "ok": "🟢",
    "degraded": "🟡",
    "error": "🔴",
}

_SCREENS = ["job", "graph", "manual", "admin"]
_SCREEN_LABELS = {
    "job": "Jobs",
    "graph": "Graph",
    "manual": "Manual Actions",
    "admin": "Admin Panel",
}


def main() -> None:
    st.set_page_config(page_title="Seshat", layout="wide", page_icon="📚")

    if "screen" not in st.session_state:
        st.session_state["screen"] = "job"

    # ── User section ──────────────────────────────────────────────────────────
    st.sidebar.subheader("Workspace")

    api_key = st.sidebar.text_input("API Key", type="password", key="api_key")

    client: ApiClient | None = None
    if api_key:
        client = ApiClient(API_BASE, api_key)
        identity = client.whoami()
        if identity:
            st.sidebar.success(f"**{identity['user_id']}** (`{identity['role']}`)")
            for screen in [s for s in _SCREENS if s != "admin"]:
                if st.sidebar.button(_SCREEN_LABELS[screen], key=f"nav_{screen}"):
                    st.session_state["screen"] = screen
                    st.rerun()

        else:
            st.sidebar.error("Invalid API key.")
            client = None

    else:
        st.sidebar.info("Enter API key to use Seshat.")

    if client:
        st.sidebar.divider()
        st.sidebar.subheader("Overview")
        with st.sidebar.expander("System Health"):
            _render_sidebar_health(client)
        with st.sidebar.expander("Recent Jobs"):
            _render_sidebar_jobs(client)

    # ── Root / admin section ───────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Admin")
    root_key = st.sidebar.text_input("Root Key", type="password", key="root_key")
    if root_key:
        root_valid = ApiClient(API_BASE, root_key).is_root_api_key_valid()
        if not root_valid:
            st.sidebar.error("Invalid root key.")

        elif root_valid and st.sidebar.button("Admin panel", key="nav_admin"):
            st.session_state["screen"] = "admin"
            st.rerun()

    else:
        st.sidebar.info("Enter root key to access admin features.")

    st.sidebar.divider()

    screen = st.session_state["screen"]

    if screen == "admin":
        if not root_key:
            st.warning("Enter the root key in the sidebar to access admin features.")
            return

        _render_admin(ApiClient(API_BASE, root_key))
        return

    if not client:
        st.warning("Enter your API key in the sidebar.")
        return

    if screen == "job":
        _render_jobs(client, mlflow_ui_base=MLFLOW_UI_BASE)
    elif screen == "graph":
        _render_kb(client)
    elif screen == "manual":
        _render_manual_actions(client)


@st.fragment
def _render_sidebar_health(client: ApiClient) -> None:
    try:
        components_health = client.get_health()
        for name, status in components_health.get("components", {}).items():
            chip = _HEALTH_CHIP.get(status, "⚪")
            st.write(f"{chip} {name}: `{status}`")
    except Exception:
        logger.exception("Health check failed")
        st.warning("Health check failed.")


@st.fragment
def _render_sidebar_jobs(client: ApiClient) -> None:
    try:
        jobs = client.list_jobs(limit=5)
    except Exception:
        logger.exception("Could not load jobs")
        st.warning("Could not load jobs.")
        return

    if not jobs:
        st.caption("No jobs yet.")
        return

    for job in jobs:
        job_id = job["job_id"]
        job_status_str = job["status"].replace("_", " ")
        if st.button(job_label(job), help=f"Job is {job_status_str}", key=f"jnav_{job_id}"):
            st.session_state["job_id"] = job_id
            st.session_state["screen"] = "job"
            st.rerun()


if __name__ == "__main__":
    main()
