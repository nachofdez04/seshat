from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    import httpx

    from seshat_ui.api_client import ApiClient

_NODE_CACHE_KEY = "_node_cache"


JOB_STATUS_CHIP: dict[str, str] = {
    "pending": "⏳",
    "transcribing": "🎙️",
    "identifying": "🔍",
    "awaiting_review": "👁️",
    "resolving": "🔗",
    "writing": "✍️",
    "done": "✅",
    "failed": "❌",
}


def job_label(job: dict) -> str:
    """Return a rich one-line label for a job dict."""
    jid = job["job_id"]
    status = job["status"]
    chip = JOB_STATUS_CHIP.get(status, "❓")
    created_at = (job.get("created_at") or "")[:10]  # only date part
    return f"{chip} `{jid[:8]}` — {created_at}"


def api_error(resp: httpx.Response) -> str:
    """Extract a human-readable error message from an API response."""
    try:
        detail = resp.json()["detail"]
        if isinstance(detail, list):
            return "; ".join(e.get("msg", str(e)) for e in detail)
        return detail
    except Exception:
        return f"({resp.status_code}) {resp.text}"


def show_api_error(resp: httpx.Response, prefix: str = "Failed") -> None:
    st.error(f"{prefix}: {api_error(resp)}")


def enrich_relationships(
    relationships: list[dict], client: ApiClient, node_index: dict[str, dict] | None = None
) -> list[dict]:
    """Enrich a list of relationship dicts with source/target node stubs (id, title, description, type)."""

    def _node_stub(node: dict) -> dict:
        return {
            "node_id": node["id"],
            "title": node["title"],
            "description": node["description"],
            "type": node["type"],
        }

    cache: dict[str, dict] = st.session_state.setdefault(_NODE_CACHE_KEY, {})

    all_ids = {r["source_id"] for r in relationships} | {r["target_id"] for r in relationships}
    missing = all_ids - cache.keys() - set(node_index or {})

    def _fetch(node_id: str) -> tuple[str, dict]:
        try:
            return node_id, _node_stub(client.get_node(node_id))
        except Exception:
            return node_id, {"node_id": node_id}

    if missing:
        with ThreadPoolExecutor(max_workers=min(len(missing), 10)) as pool:
            for node_id, stub in pool.map(_fetch, missing):
                cache[node_id] = stub

    lookup: dict[str, dict] = {}
    for node_id in all_ids:
        if node_index and node_id in node_index:
            lookup[node_id] = _node_stub(node_index[node_id])
        else:
            lookup[node_id] = cache[node_id]

    return [
        {
            "relationship_id": r["rel_id"],
            "source": lookup[r["source_id"]],
            "target": lookup[r["target_id"]],
            "relationship_type": r["rel_type"],
        }
        for r in relationships
    ]
