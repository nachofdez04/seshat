from __future__ import annotations

import sys
from collections import Counter
from functools import reduce
from typing import TYPE_CHECKING

import streamlit as st

from seshat_ui.utils import enrich_relationships, job_label

if TYPE_CHECKING:
    from collections.abc import Iterable

    from seshat_ui.api_client import ApiClient

_IN_PROGRESS_STATUSES = ("pending", "transcribing", "identifying", "resolving", "writing")
_STAGES = ["pending", "transcribing", "identifying", "awaiting_review", "resolving", "writing", "done"]
_STAGE_LABELS = {
    "pending": "Queued — waiting to start...",
    "transcribing": "Transcribing audio...",
    "identifying": "Identifying knowledge nodes...",
    "awaiting_review": "Awaiting review...",
    "resolving": "Resolving node relationships...",
    "writing": "Writing to knowledge base...",
    "done": "Complete",
}

_FILE_TYPES = {
    "audio": ["mp3", "wav", "m4a"],
    "text": ["yaml", "json"],
}
_ALL_FILE_TYPES = reduce(list.__add__, _FILE_TYPES.values())

_SOURCE_TYPES = ["all", "audio", "text"]
_JOB_STATUSES = [
    "all",
    "pending",
    "transcribing",
    "identifying",
    "awaiting_review",
    "resolving",
    "writing",
    "done",
    "failed",
]
_PAGE_SIZE = 25


def render(client: ApiClient, mlflow_ui_base: str = "") -> None:
    st.title("Jobs")

    job_id = st.session_state.get("job_id")

    if not job_id:
        submit_tab, list_tab = st.tabs(["📤 Submit", "🔍 Search jobs"])
        with submit_tab:
            _render_submit(client)
        with list_tab:
            _render_job_search(client)
        return

    try:
        job = client.get_job(job_id)
    except Exception as exc:
        st.error(f"Could not fetch job: {exc}")
        if st.button("Start new job"):
            st.session_state.pop("job_id", None)
            st.rerun()
        return

    status = job["status"]

    if status in _IN_PROGRESS_STATUSES:
        _render_progress(client, job_id)
    elif status == "awaiting_review":
        _render_review(client, job)
    elif status == "done":
        _render_summary(client, job_id, mlflow_ui_base=mlflow_ui_base)
    elif status == "failed":
        err = job.get("error") or {}
        st.error(f"Job failed at stage `{err.get('stage', '?')}`: {err.get('reason', 'unknown error')}")
        if st.button("Retry job"):
            resp = client.retry_job(job_id)
            if resp.is_success:
                st.rerun()
            else:
                st.error(f"Retry failed: {resp.text}")
        if st.button("Start new job"):
            st.session_state.pop("job_id", None)
            st.rerun()


def _render_submit(client: ApiClient) -> None:
    st.header("Submit Meeting")
    with st.form("submit_job_form"):
        col1, col2, col3, col4 = st.columns([0.3, 0.3, 0.25, 0.15])
        meeting_date = col1.date_input("Meeting date", key="s_date")
        source_type = col2.selectbox("Source type", ["text", "audio"], key="s_source_type")
        confidence_threshold_raw = col3.slider(
            "Confidence threshold",
            0.0,
            1.05,
            0.7,
            0.05,
            help=(
                "Nodes that meet or exceed this score (and pass grounding) are auto-approved. "
                "Nodes below it go to manual review in normal mode, or are rejected outright in auto mode.\n"
                "Set to 1.05 to disable auto-approval entirely (all nodes go to manual review)."
            ),
            key="s_conf_threshold",
        )
        auto_mode = col4.toggle(
            "Auto mode",
            value=False,
            help=(
                "When on, nodes below the confidence threshold are rejected automatically — no human review step. "
                "When off, those nodes are queued for manual review instead."
            ),
            width="stretch",
            key="s_auto_mode",
        )
        force_ingest = col4.toggle(
            "Force re-ingest",
            value=False,
            help=(
                "When on, ALL existing KB nodes from the previous ingest of this file are permanently deleted, "
                "then the file is re-ingested from scratch. This is irreversible and requires admin role.\n"
                "When off, the job is rejected if the file was already ingested."
            ),
            width="stretch",
            key="s_force_ingest",
        )
        participants_raw = st.text_input("Participants (comma-separated, optional)", key="s_participants")
        uploaded = st.file_uploader("Upload file", type=_ALL_FILE_TYPES, key="s_file")
        submitted = st.form_submit_button("Submit")

    if submitted:
        if not uploaded:
            st.error("Please upload a file.")
            return

        uploaded_filename = uploaded.name
        ext = uploaded_filename.split(".")[-1].lower()
        if ext not in _FILE_TYPES.get(source_type, []):
            st.error(f"Invalid file type for {source_type!r} source type: {ext}")
            return

        confidence_threshold: float | None = None if confidence_threshold_raw > 1.0 else confidence_threshold_raw
        participants = [p.strip() for p in participants_raw.split(",") if p.strip()] or None
        body: dict = {
            "source_type": source_type,
            "metadata": {"meeting_date": meeting_date.isoformat(), "participants": participants},
            "overrides": {"extraction": {"confidence_threshold": confidence_threshold}},
            "force": force_ingest,
            "auto_mode": auto_mode,
        }
        resp = client.submit_job(uploaded.read(), uploaded_filename, body)

        if resp.is_success:
            job_id = resp.json()["job_id"]
            st.success(f"Job {job_id[:8]!r} submitted successfully. You will be redirected to the job status page.")
            st.session_state["job_id"] = job_id
            st.rerun()
        elif resp.status_code == 409:
            data = resp.json()
            st.warning(f"Already ingested as job `{data.get('existing_job_id')}`. Use `force=true` to re-ingest.")
        else:
            st.error(f"Submission failed ({resp.status_code}): {resp.text}")


def _render_summary(client: ApiClient, job_id: str, mlflow_ui_base: str = "") -> None:
    st.success(f"Job {job_id[:8]} completed.")

    try:
        job = client.get_job(job_id)
        result = client.get_job_results(job_id)
    except Exception:
        st.warning("Could not load result details.")
        return

    mlflow_run_id = job.get("mlflow_run_id")
    if mlflow_run_id:
        if mlflow_ui_base:
            url = f"{mlflow_ui_base.rstrip('/')}/#/runs/{mlflow_run_id}"
            st.caption(f"MLflow run: [{mlflow_run_id}]({url})")
        else:
            st.caption(f"MLflow run: `{mlflow_run_id}`")

    nodes = result.get("nodes", [])

    approved = [n for n in nodes if n["status"] == "approved"]
    rejected = [n for n in nodes if n["status"] == "rejected"]
    relationships = result.get("relationships", [])

    st.subheader("Job results summary")
    cols = st.columns(3)
    cols[0].metric("Approved", len(approved), help=_type_breakdown(n["type"] for n in approved))
    cols[1].metric("Rejected", len(rejected), help=_type_breakdown(n["type"] for n in rejected))
    cols[2].metric("Relationships", len(relationships), help=_type_breakdown(r["rel_type"] for r in relationships))

    st.divider()

    st.subheader("Job details")

    with st.expander(f"Approved nodes ({len(approved)})", expanded=False):
        st.json(approved)

    if rejected:
        with st.expander(f"Rejected nodes ({len(rejected)})", expanded=False):
            st.json(rejected)

    with st.expander(f"Relationships ({len(relationships)})", expanded=False):
        node_index = {n["id"]: n for n in nodes}
        enriched_rels = enrich_relationships(relationships, client, node_index)
        st.json(enriched_rels)

    st.divider()

    col1, col2, _ = st.columns([1, 1, 4])
    if col1.button("View KB →"):
        st.session_state["screen"] = "graph"
        st.rerun()
    if col2.button("Submit another job"):
        st.session_state.pop("job_id", None)
        st.rerun()


def _render_job_search(client: ApiClient) -> None:
    with st.form("job_search_form"):
        col1, col2 = st.columns(2)
        status_filter = col1.selectbox("Status", _JOB_STATUSES, format_func=lambda x: x.capitalize(), key="js_status")
        source_filter = col2.selectbox(
            "Source type", _SOURCE_TYPES, format_func=lambda x: x.capitalize(), key="js_source"
        )

        col3, col4 = st.columns(2)
        date_from = col3.date_input("Meeting date from", value=None, key="js_date_from")
        date_to = col4.date_input("Meeting date to", value=None, key="js_date_to")

        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        st.session_state["js_offset"] = 0

    offset = st.session_state.get("js_offset", 0)

    if submitted:
        filters = {
            "status": status_filter if status_filter != "all" else None,
            "source_type": source_filter if source_filter != "all" else None,
            "meeting_date_from": str(date_from) if date_from else None,
            "meeting_date_to": str(date_to) if date_to else None,
        }
        try:
            jobs = client.list_jobs(**filters, limit=_PAGE_SIZE, offset=0)
        except Exception as exc:
            st.error(f"Could not load jobs: {exc}")
            return
        st.session_state["js_results"] = jobs
        st.session_state["js_filters"] = filters
        _render_job_results(client, jobs, 0)
    elif "js_results" in st.session_state:
        _render_job_results(client, st.session_state["js_results"], offset)


def _render_job_results(client: ApiClient, jobs: list[dict], offset: int) -> None:
    if not jobs and offset == 0:
        st.info("No jobs found.")
        return

    for job in jobs:
        jid = job["job_id"]
        if st.button(job_label(job), key=f"jlist_{jid}"):
            st.session_state["job_id"] = jid
            st.rerun()

    col_prev, col_next = st.columns([1, 1])
    filters = st.session_state.get("js_filters", {})
    if offset > 0 and col_prev.button("← Previous page"):
        new_offset = max(0, offset - _PAGE_SIZE)
        st.session_state["js_offset"] = new_offset
        try:
            st.session_state["js_results"] = client.list_jobs(**filters, limit=_PAGE_SIZE, offset=new_offset)
        except Exception as exc:
            st.error(f"Could not load jobs: {exc}")
        st.rerun()

    if len(jobs) == _PAGE_SIZE and col_next.button("Next page →"):
        new_offset = offset + _PAGE_SIZE
        st.session_state["js_offset"] = new_offset
        try:
            st.session_state["js_results"] = client.list_jobs(**filters, limit=_PAGE_SIZE, offset=new_offset)
        except Exception as exc:
            st.error(f"Could not load jobs: {exc}")
        st.rerun()


@st.fragment(run_every=3)
def _render_progress(client: ApiClient, job_id: str) -> None:
    try:
        job = client.get_job(job_id)
    except Exception:
        st.warning("Could not refresh job status.")
        return

    status = job["status"]
    if status not in _IN_PROGRESS_STATUSES:
        st.rerun(scope="app")
        return

    st.header(f"Processing — {job_id[:8]}")

    idx = _STAGES.index(status) if status in _STAGES else 0
    label = _STAGE_LABELS.get(status, status.replace("_", " ").title())
    st.progress((idx + 1) / len(_STAGES), text=label)

    elapsed = job.get("elapsed_seconds")
    if elapsed is not None:
        st.caption(f"Elapsed: {elapsed:.1f}s")


@st.fragment
def _render_review(client: ApiClient, job: dict) -> None:
    job_id = job["job_id"]
    st.header(f"Review — {job_id[:8]}")

    try:
        result = client.get_job_results(job_id)
    except Exception as exc:
        st.error(f"Could not load results: {exc}")
        return

    nodes = result.get("nodes", [])
    breakdowns = result.get("confidence_breakdowns", {})
    pending = [n for n in nodes if n["status"] == "pending_review"]

    if not pending:
        st.info("No nodes pending review — pipeline will complete automatically.")
        st.rerun(scope="app")
        return

    st.write(f"**{len(pending)} nodes pending review**")

    with st.expander("Full transcript", expanded=False):
        transcript = client.get_transcript_excerpt(job_id, 0, sys.maxsize)
        if transcript:
            st.text(transcript)
        else:
            st.caption("Transcript not available.")

    st.subheader("Bulk decision options")
    use_bulk = st.toggle(
        "Enable bulk-approve rules",
        value=False,
        help="Bulk-approve rules are those applied to several nodes at once",
        key="r_use_bulk",
    )

    bulk_rule: dict | None = None
    if use_bulk:
        threshold = st.slider(
            "Auto-approve above threshold",
            0.5,
            1.0,
            0.7,
            0.05,
            help=(
                "Nodes with confidence at or above this threshold are approved automatically.\n"
                "Nodes below it remain pending and are resolved by your individual decisions below."
            ),
            key="r_threshold",
        )
        bulk_rule = {"threshold": threshold}

    expand_threshold = job.get("confidence_threshold") or 1.0

    st.subheader("Individual node review")
    decisions: list[dict] = []
    for node in pending:
        node_id = node["id"]
        confidence_breakdown = breakdowns.get(node_id, {})
        node_confidence = node.get("confidence", 0.0)
        label = f"[{node['type'].upper()}] {node['title']} (Confidence={node_confidence:.2f})"
        expanded_flag = node_confidence < expand_threshold

        with st.expander(label, expanded=expanded_flag):
            st.write(f"**Description:** {node['description']}")

            anchors = node.get("quote_anchors", [])
            if anchors:
                a = anchors[0]
                excerpt = client.get_transcript_excerpt(job_id, a["char_start"], a["char_end"])
                if excerpt:
                    st.info(f"**Source:** […] {excerpt} […]")
                else:
                    st.caption(f"Quote anchor: chars {a['char_start']}-{a['char_end']}")

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Heuristics",
                f"{confidence_breakdown.get('heuristics', 0):.2f}",
                help=(
                    "Rule-based score [0-1] derived from signal density, specificity, "
                    "and structure. Main auto-approval signal."
                ),
            )
            c2.metric(
                "Grounding",
                _get_grounding_val(confidence_breakdown),
                help=(
                    "Whether the node's claim is supported by a verbatim quote in the transcript. "
                    "'—' means grounding was not run."
                ),
            )
            c3.metric(
                "Confidence",
                f"{node_confidence:.2f}",
                help=(
                    "Final score used for auto-approval. "
                    "Equals heuristics when grounding is disabled; grounding failure caps it at 0."
                ),
            )

            pending_reason = node.get("metadata", {}).get("pending_reason")
            if pending_reason:
                st.warning(f"**Pending reason**: {pending_reason}")

            action = st.radio(
                "Decision",
                ["approve", "reject", "edit"],
                key=f"r_action_{node_id}",
                horizontal=True,
            )
            edited = None
            if action == "edit":
                new_title = st.text_input("Title", value=node["title"], key=f"r_title_{node_id}")
                new_desc = st.text_area("Description", value=node["description"], key=f"r_desc_{node_id}")
                edited = {"title": new_title, "description": new_desc}

            decisions.append(
                {
                    "node_id": node_id,
                    "action": "approve" if action != "reject" else "reject",
                    "edited_content": edited,
                }
            )

    if st.button("Submit decisions", key="r_submit"):
        payload: dict = {"decisions": decisions}
        if bulk_rule:
            payload["approve_above_threshold"] = bulk_rule
        resp = client.approve_job(job_id, payload)
        if resp.is_success:
            st.success("Decisions submitted.")
            st.rerun(scope="app")
        else:
            st.error(f"Failed: {resp.text}")


def _type_breakdown(types: Iterable[str]) -> str:
    counts = Counter(types)
    return "  ·  ".join(f"{t}: {c}" for t, c in sorted(counts.items()))


def _get_grounding_val(confidence_breakdown: dict) -> str:
    if confidence_breakdown.get("grounding_passed"):
        return "✓"
    elif confidence_breakdown.get("grounding_enabled"):
        return "✗"
    else:
        return "—"
