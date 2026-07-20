from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import streamlit as st

from seshat_ui.utils import api_error, enrich_relationships, show_api_error

if TYPE_CHECKING:
    from seshat_ui.api_client import ApiClient

_NODE_TYPES = ["decision", "risk", "action_item", "open_question"]
_REL_TYPES = ["mitigates", "blocks", "conflicts_with", "depends_on", "supersedes", "amends", "resolves"]


def _load_node(client: ApiClient, node_key: str, id_key: str) -> None:
    node_id = st.session_state.get(id_key, "").strip()
    st.session_state.pop(node_key, None)
    st.session_state.pop(f"{node_key}_err", None)
    if not node_id:
        return
    try:
        st.session_state[node_key] = client.get_node(node_id)
    except httpx.HTTPStatusError as exc:
        st.session_state[f"{node_key}_err"] = api_error(exc.response)
    except Exception as exc:
        st.session_state[f"{node_key}_err"] = str(exc)


def render(client: ApiClient) -> None:
    st.title("Manual Actions")
    nodes_section, rels_section = st.tabs(["🔵 Nodes", "🔗 Relationships"])

    with nodes_section:
        create_tab, edit_tab, override_tab, delete_tab = st.tabs(["➕ Create", "✏️ Edit", "🔄 Override", "🗑️ Delete"])  # noqa: RUF001

        with create_tab:
            _render_create(client)

        with edit_tab:
            _render_edit(client)

        with override_tab:
            _render_override(client)

        with delete_tab:
            _render_delete(client)

    with rels_section:
        list_tab, create_rel_tab, delete_rel_tab = st.tabs(["🔍 List", "➕ Create", "🗑️ Delete"])  # noqa: RUF001

        with list_tab:
            _render_list_relationships(client)

        with create_rel_tab:
            _render_create_relationship(client)

        with delete_rel_tab:
            _render_delete_relationship(client)


@st.fragment
def _render_create(client: ApiClient) -> None:
    st.caption("Create a new node. Optionally define relationships to other nodes, or let the system infer them.")
    node_type = st.selectbox(
        "Type", _NODE_TYPES, format_func=lambda x: x.replace("_", " ").capitalize(), key="m_c_type"
    )
    title = st.text_input("Title", key="m_c_title")
    description = st.text_area("Description", key="m_c_desc")

    col1, col2, col3 = st.columns(3)
    meeting_date = col1.date_input("Meeting date", key="m_c_date")
    team = col2.text_input(
        "Team", placeholder="optional", help="Team is stored as optional node metadata", key="m_c_team"
    )
    project = col3.text_input(
        "Project", placeholder="optional", help="Project is stored as optional node metadata", key="m_c_project"
    )

    col1, col2 = st.columns([0.85, 0.15])
    auto_resolve = col2.toggle(
        "Auto-resolve relationships",
        value=True,
        help=(
            "When on, the node is passed through the resolution pipeline after creation — "
            "relationships are inferred automatically by the LLM.\n"
            "When off, you can define relationships manually below."
        ),
        key="m_c_auto_resolve",
    )

    with col1.expander("Manual Relationships", expanded=(not auto_resolve)):
        rels: list[dict] = st.session_state.setdefault("m_c_rels", [])
        for i, rel in enumerate(rels):
            col1, col2, col3 = st.columns([2, 1, 0.3])
            rels[i]["target_id"] = col1.text_input("Target node ID", value=rel["target_id"], key=f"m_c_rel_tid_{i}")
            rels[i]["rel_type"] = col2.selectbox(
                "Type",
                _REL_TYPES,
                index=_REL_TYPES.index(rel["rel_type"]),
                format_func=lambda x: x.replace("_", " ").capitalize(),
                key=f"m_c_rel_rtype_{i}",
            )
            if col3.button("✕", key=f"m_c_rel_rm_{i}"):
                rels.pop(i)
                st.rerun()

        help_msg = (
            "Disabled when auto-resolve is on. Relationships will be inferred automatically by the LLM."
            if auto_resolve
            else None
        )
        if st.button("＋ Add relationship", disabled=auto_resolve, help=help_msg, key="m_c_rel_add"):  # noqa: RUF001
            rels.append({"target_id": "", "rel_type": _REL_TYPES[0]})
            st.rerun()

    if st.button("Create", type="primary", key="m_c_submit"):
        if not title or not description:
            st.error("Title and description are required.")
            return

        payload: dict = {
            "type": node_type,
            "title": title,
            "description": description,
            "meeting_date": meeting_date.isoformat(),
        }
        if team:
            payload["team"] = team
        if project:
            payload["project"] = project

        if not auto_resolve:
            valid_rels = [r for r in st.session_state.get("m_c_rels", []) if r["target_id"].strip()]
            if valid_rels:
                payload["relationships"] = valid_rels

        resp = client.create_node(payload)
        if not resp.is_success:
            show_api_error(resp)
            return

        node_id = resp.json()["id"]
        st.success(f"Node created: `{node_id}`")
        st.session_state["m_c_rels"] = []

        if auto_resolve:
            with st.spinner("Resolving relationships..."):
                resolve_resp = client.resolve_nodes([node_id])

            if resolve_resp.is_success:
                rels = resolve_resp.json().get("relationships_created", [])
                st.info(f"Resolution complete: {len(rels)} relationship(s) created.")
                if rels:
                    enriched = enrich_relationships(rels, client)
                    with st.expander("Resolved relationships detail", expanded=False):
                        st.json(enriched)
            else:
                show_api_error(resolve_resp, prefix="Resolution failed")


def _render_edit(client: ApiClient) -> None:
    st.caption("Alters a manually-created node. Use the `Override` tab to modify pipeline-generated nodes.")
    st.text_input(
        "Node ID",
        key="m_e_id",
        on_change=_load_node,
        args=(client, "m_e_node", "m_e_id"),
    )

    if err := st.session_state.get("m_e_node_err"):
        st.error(err)

    node = st.session_state.get("m_e_node")
    if not node:
        return

    new_title = st.text_input("Title", value=node["title"], key="m_e_title")
    new_desc = st.text_area("Description", value=node["description"], key="m_e_desc")
    reason = st.text_input("Reason", placeholder="Optional edit reason", key="m_e_reason")

    if st.button("Save", key="m_e_submit"):
        payload = {"title": new_title, "description": new_desc, "reason": reason or None}
        resp = client.update_node(st.session_state["m_e_id"], payload)
        if resp.is_success:
            st.success("Node updated.")
            st.session_state.pop("m_e_node", None)
        else:
            show_api_error(resp)


def _render_override(client: ApiClient) -> None:
    st.caption("Alters a pipeline-generated node. Records the correcting user and reason in the node metadata.")
    st.text_input(
        "Node ID",
        key="m_o_id",
        on_change=_load_node,
        args=(client, "m_o_node", "m_o_id"),
    )

    if err := st.session_state.get("m_o_node_err"):
        st.error(err)

    node = st.session_state.get("m_o_node")
    if not node:
        return

    new_title = st.text_input("Title", value=node["title"], key="m_o_title")
    new_desc = st.text_area("Description", value=node["description"], key="m_o_desc")
    reason = st.text_input("Reason", placeholder="Required edit reason", key="m_o_reason")

    if st.button("Override", key="m_o_submit"):
        if not reason:
            st.error("Reason is required for overrides.")
            return

        payload = {"title": new_title, "description": new_desc, "reason": reason}
        resp = client.override_node(st.session_state["m_o_id"], payload)
        if resp.is_success:
            st.success(f"Override created: `{resp.json()['id']}`")
            st.session_state.pop("m_o_node", None)
        else:
            show_api_error(resp)


def _render_delete(client: ApiClient) -> None:
    st.caption("Permanently removes a node from the KB. **This is irreversible**.")
    st.text_input(
        "Node ID",
        key="m_d_id",
        on_change=_load_node,
        args=(client, "m_d_node", "m_d_id"),
    )

    if err := st.session_state.get("m_d_node_err"):
        st.error(err)

    node = st.session_state.get("m_d_node")
    if not node:
        return

    with st.expander("Node detail", expanded=False):
        st.json(node)

    cascade = st.toggle(
        "Cascade",
        value=True,
        help=(
            "When on, also deletes all relationships connected to this node. "
            "When off, deletion fails if any relationships exist."
        ),
        key="m_d_cascade",
    )
    st.warning("This will permanently delete the node" + (" and all its relationships." if cascade else "."))
    if st.button("Delete", type="primary", key="m_d_submit"):
        resp = client.delete_node(st.session_state["m_d_id"], cascade=cascade)
        if resp.is_success:
            st.success("Node deleted.")
            st.session_state.pop("m_d_node", None)
        else:
            show_api_error(resp)


@st.fragment
def _render_list_relationships(client: ApiClient) -> None:
    st.caption("Browse relationships, optionally filtered by node ID or type.")
    col1, col2, col3 = st.columns([2, 1, 0.5])
    node_id = col1.text_input("Node ID", placeholder="optional", key="m_lr_node_id")
    rel_type = col2.selectbox(
        "Type", ["all", *_REL_TYPES], format_func=lambda x: x.replace("_", " ").capitalize(), key="m_lr_rel_type"
    )
    limit = col3.number_input("Limit", min_value=1, max_value=1000, value=50, key="m_lr_limit")

    if st.button("Search", key="m_lr_submit"):
        rels = client.list_relationships(
            node_id=node_id.strip() or None,
            rel_type=None if rel_type == "all" else rel_type,
            limit=int(limit),
        )
        with st.spinner("Fetching relationship details..."):
            enriched = enrich_relationships(rels, client)
            st.session_state["m_lr_results"] = enriched

    results = st.session_state.get("m_lr_results")
    if results is None:
        return

    st.info(f"{len(results)} relationship(s) found.")
    for relationship_data in results:
        src = relationship_data["source"]
        tgt = relationship_data["target"]

        src_str = src.get("title", src["node_id"])
        tgt_str = tgt.get("title", tgt["node_id"])
        label = f"**{src_str}** → `{relationship_data['relationship_type']}` → **{tgt_str}**"
        with st.expander(label, expanded=False):
            st.json(relationship_data)


@st.fragment
def _render_create_relationship(client: ApiClient) -> None:
    st.caption("Create a standalone relationship between two existing nodes.")
    source_id = st.text_input("Source node ID", key="m_cr_source")
    target_id = st.text_input("Target node ID", key="m_cr_target")
    rel_type = st.selectbox(
        "Relationship type", _REL_TYPES, format_func=lambda x: x.replace("_", " ").capitalize(), key="m_cr_rel_type"
    )

    if st.button("Create", type="primary", key="m_cr_submit"):
        if not source_id.strip() or not target_id.strip():
            st.error("Source and target node IDs are required.")
            return

        resp = client.create_relationship(source_id.strip(), target_id.strip(), rel_type)
        if resp.is_success:
            rel = resp.json()
            st.success(f"Relationship created: `{rel['rel_id']}`")
        else:
            show_api_error(resp)


@st.fragment
def _render_delete_relationship(client: ApiClient) -> None:
    st.caption("Permanently removes a relationship by its ID. **This is irreversible**.")
    rel_id = st.text_input("Relationship ID", key="m_dr_rel_id")

    if rel_id.strip():
        st.warning("This will permanently delete the relationship.")

    if st.button("Delete", type="primary", key="m_dr_submit"):
        if not rel_id.strip():
            st.error("Relationship ID is required.")
            return

        resp = client.delete_relationship(rel_id.strip())
        if resp.is_success:
            st.success("Relationship deleted.")
        elif resp.status_code == 404:
            st.error(f"Relationship not found: `{rel_id.strip()}`")
        else:
            show_api_error(resp)
