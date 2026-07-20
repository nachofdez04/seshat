from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

if TYPE_CHECKING:
    from seshat_ui.api_client import ApiClient

_NODE_TYPES = ["all", "decision", "risk", "action_item", "open_question"]
_INGESTION_SOURCES = ["all", "pipeline", "manual"]
_SEARCH_MODES = ["semantic", "keyword", "hybrid"]
_STATE_BADGE: dict[str, str] = {"current": "🟢", "amended": "🟡", "superseded": "🔴"}


def render(client: ApiClient) -> None:
    st.title("Graph")
    browse_tab, search_tab, impact_tab = st.tabs(["📋 Browse", "🔍 Search", "🌐 Impact"])

    with browse_tab:
        _render_browse(client)

    with search_tab:
        _render_search(client)

    with impact_tab:
        _render_impact(client)


def _node_filter_widgets(prefix: str) -> dict:
    """Render the shared node filter controls inside the caller's form. Returns a filters dict."""
    col1, col2, col3, col4 = st.columns(4)
    node_type = col1.selectbox(
        "Node type", _NODE_TYPES, format_func=lambda x: x.replace("_", " ").capitalize(), key=f"{prefix}_type"
    )
    source = col2.selectbox(
        "Ingestion source", _INGESTION_SOURCES, format_func=lambda x: x.capitalize(), key=f"{prefix}_source"
    )
    date_from = col3.date_input("Meeting date from", value=None, key=f"{prefix}_date_from")
    date_to = col4.date_input("Meeting date to", value=None, key=f"{prefix}_date_to")

    filters: dict = {}
    if node_type and node_type != "all":
        filters["node_type"] = node_type
    if source and source != "all":
        filters["ingestion_source"] = source
    if date_from:
        filters["meeting_date_from"] = str(date_from)
    if date_to:
        filters["meeting_date_to"] = str(date_to)
    return filters


_NODE_STATES = ["all", "current", "amended", "superseded"]


def _render_browse(client: ApiClient) -> None:
    st.caption("Filter nodes by structured attributes. Results come from the knowledge store.")
    with st.form("kb_browse_form"):
        filters = _node_filter_widgets("kb_b")

        col1, col2, col3, col4 = st.columns(4)
        state = col1.selectbox("Node state", _NODE_STATES, format_func=lambda x: x.capitalize(), key="kb_b_state")
        job_id = col2.text_input("Job ID", placeholder="Job UUID", key="kb_b_job_id")
        min_conf = col3.slider(
            "Min confidence", 0.0, 1.0, 0.0, 0.05, help="Minimum node grounding confidence", key="kb_b_conf"
        )
        limit = col4.number_input(
            "Limit", min_value=1, max_value=10000, value=100, help="Maximum number of displayed nodes", key="kb_b_limit"
        )

        submitted = st.form_submit_button("Browse")

    if submitted:
        filters["limit"] = limit
        if state and state != "all":
            filters["state"] = state
        if job_id:
            filters["job_id"] = job_id
        if min_conf > 0.0:
            filters["min_confidence"] = min_conf

        try:
            st.session_state["kb_b_nodes"] = client.query_graph(**filters)
            st.session_state.pop("b_selected_node", None)
        except Exception as exc:
            st.error(f"Failed to query KB: {exc}")
            return

    nodes = st.session_state.get("kb_b_nodes")
    if nodes is not None:
        _render_node_list(nodes, key_prefix="b")


def _render_search(client: ApiClient) -> None:
    st.caption("Semantic, keyword, or hybrid search over the vector store. Filters apply at the embedding layer.")
    with st.form("kb_search_form"):
        query = st.text_input(
            "Search query",
            placeholder="e.g. authentication risks from last quarter",
            key="kb_s_query",
        )

        col1, col2 = st.columns(2)
        limit = col1.number_input("Max results", 1, 50, 10, key="kb_s_limit")
        search_mode = col2.selectbox("Search mode", _SEARCH_MODES, key="kb_s_mode")

        with st.expander("Filters", expanded=False):
            filters = _node_filter_widgets("kb_s")

        submitted = st.form_submit_button("Search", type="primary")

    results: list[dict] | None
    if submitted:
        if not query:
            st.warning("Enter a search query.")
        else:
            try:
                results = client.search_graph(query, limit=limit, search_mode=search_mode, **filters)
                st.session_state["kb_s_results"] = results
                st.session_state.pop("s_selected_node", None)
            except Exception as exc:
                st.error(f"Search failed: {exc}")
                return

    results = st.session_state.get("kb_s_results")
    if results is not None:
        _render_search_results(results, key_prefix="s")


def _render_search_results(results: list[dict], key_prefix: str) -> None:
    nodes = [r["detail"]["node"] for r in results]
    scores = {r["detail"]["node"]["id"]: r.get("score") for r in results}
    _render_node_list(nodes, key_prefix=key_prefix, scores=scores)


def _render_node_list(nodes: list[dict], key_prefix: str, scores: dict[str, float | None] | None = None) -> None:
    if not nodes:
        st.info("No nodes found.")
        return

    st.write(f"**{len(nodes)} nodes**")

    for node in nodes:
        badge = _STATE_BADGE.get(node.get("state", ""), "⚪")
        conf = node.get("confidence", 0.0)
        score = (scores or {}).get(node["id"])
        score_str = f", Score={score:.3f}" if score is not None else ""
        label = f"{badge} [{node['type'].upper()}] {node['title']} (Confidence={conf:.2f}{score_str})"

        with st.expander(label):
            st.write(node["description"])

            meta = node["metadata"]
            st.caption(
                f"**ID:** `{node['id']}`  \n"
                f"**Job ID:** `{meta['job_id']}`  \n"
                f"**State:** `{node['state'].upper()}`  \n"
                f"**Meeting date:** `{meta['meeting_date']}`"
            )

            anchors = node.get("quote_anchors", [])
            if anchors:
                a = anchors[0]
                st.caption(f"Quote anchor: characters {a['char_start']}-{a['char_end']}")

            node_id = node["id"]
            if st.button("Load impact →", key=f"{key_prefix}_detail_{node_id}"):
                st.session_state["kb_impact_node"] = node_id
                st.session_state["kb_impact_params"] = {
                    "node_id": node_id,
                    "depth": 2,
                    "rel_types": None,
                    "min_confidence": 0.0,
                }
                st.toast("Impact loaded — switch to the 🌐 Impact tab")


_REL_TYPE_OPTIONS = ["mitigates", "blocks", "conflicts_with", "depends_on", "supersedes", "amends", "resolves"]


@st.fragment
def _render_impact(client: ApiClient) -> None:
    st.caption("Traverse the impact graph for any node.")
    with st.form("kb_impact_form"):
        col1, col2 = st.columns([2 / 3, 1 / 3])
        node_id_input = col1.text_input(
            "Node ID",
            value=st.session_state.get("kb_impact_node", ""),
            placeholder="Paste a node UUID or select one from Browse / Search",
        )
        direction = col2.selectbox(
            "Direction",
            ["outbound", "inbound"],
            key="kb_i_direction",
            help=(
                "**Outbound** — nodes this node affects downstream (e.g. what does this decision impact?).\n "
                "**Inbound** — nodes that shaped this node upstream (e.g. what influenced this decision?)."
            ),
        )

        col3, col4, col5 = st.columns(3)
        depth = col3.number_input(
            "Traversal depth",
            min_value=1,
            max_value=3,
            value=2,
            step=1,
            key="kb_i_depth",
            help="How many hops to follow. Higher values show more distant nodes but may be noisy.",
        )
        rel_types = col4.multiselect(
            "Relationship types",
            _REL_TYPE_OPTIONS,
            default=[],
            key="kb_i_rel_types",
            help="Restrict traversal to these relationship types. Leave empty to follow all types.",
        )
        min_confidence = col5.slider(
            "Min confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="kb_i_min_conf",
            help="Exclude nodes whose confidence score is below this threshold.",
        )

        submitted = st.form_submit_button("Load", type="primary")

    if submitted and node_id_input:
        st.session_state["kb_impact_node"] = node_id_input
        st.session_state["kb_impact_params"] = {
            "node_id": node_id_input,
            "depth": depth,
            "direction": direction,
            "rel_types": rel_types or None,
            "min_confidence": min_confidence,
        }

    params = st.session_state.get("kb_impact_params")
    if not params:
        st.info("Enter a node ID or select one from Browse or Search.")
        return

    _render_node_detail(
        client,
        params["node_id"],
        depth=params["depth"],
        direction=params.get("direction", "outbound"),
        rel_types=params["rel_types"],
        min_confidence=params["min_confidence"],
    )


def _render_node_detail(
    client: ApiClient,
    node_id: str,
    *,
    depth: int = 2,
    direction: str = "outbound",
    rel_types: list[str] | None = None,
    min_confidence: float = 0.0,
) -> None:
    try:
        detail = client.get_node_detail(node_id)
    except Exception as exc:
        st.error(f"Could not load node detail: {exc}")
        return

    source_node = detail["node"]
    st.subheader("Current node")
    st.write(f"**[This node]:**  {source_node['title']}  ({source_node['type'].upper()})")
    meta = source_node.get("metadata", {})
    st.caption(
        f"**ID:** `{node_id}`  \n"
        f"**Job ID:** `{meta.get('job_id', 'n/a')}`  \n"
        f"**State:** `{source_node.get('state', '').upper()}`  \n"
        f"**Meeting date:** `{meta.get('meeting_date', 'n/a')}`"
    )
    with st.expander("Full node detail", expanded=False):
        st.json(source_node)

    neighbours = detail.get("neighbours", [])
    relationships = detail.get("relationships", [])
    if neighbours:
        st.subheader("Neighbour nodes")
        rel_index: dict[str, dict] = {}
        for r in relationships:
            if r["source_id"] == node_id:
                rel_index[r["target_id"]] = {"rel_type": r["rel_type"], "direction": "outbound"}
            else:
                rel_index[r["source_id"]] = {"rel_type": r["rel_type"], "direction": "inbound"}

        with st.expander(f"**{len(neighbours)} neighbour node(s)**"):
            for n in neighbours:
                rel = rel_index.get(n["id"], {})
                rel_type = rel.get("rel_type", "?")
                ntype = n["type"].upper()
                ntitle = n["title"]
                if rel.get("direction") == "outbound":
                    st.write(f"  - **[This node]** `{rel_type}` → {ntitle} ({ntype})")
                else:
                    st.write(f"  -  {ntitle} ({ntype}) `{rel_type}` → **[This node]**")

    rel_types_str = ",".join(rel_types) if rel_types else None
    impact = client.get_node_impact(
        node_id, depth=depth, direction=direction, rel_types=rel_types_str, min_confidence=min_confidence
    )
    impact_nodes = impact.get("nodes", [])
    impact_rels = impact.get("relationships", [])

    st.subheader("Node impact traversal")
    if not impact_nodes:
        st.info("No impact nodes found for this node with the current filters.")
        return

    st.caption("Click a node in the graph to re-centre on it.")

    fig = _build_impact_plotly(node_id, detail["node"], impact_nodes, impact_rels)
    event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", key=f"impact_graph_{node_id}")
    pts = (event["selection"].get("points") or []) if event and event.get("selection") else []
    if pts:
        clicked_id = pts[0].get("customdata")
        if clicked_id:
            current = st.session_state.get("kb_impact_params", {})
            st.session_state["kb_impact_node"] = clicked_id
            st.session_state["kb_impact_params"] = {
                "node_id": clicked_id,
                "depth": current.get("depth", 2),
                "direction": current.get("direction", "outbound"),
                "rel_types": current.get("rel_types"),
                "min_confidence": current.get("min_confidence", 0.0),
            }
            st.rerun(scope="fragment")


def _build_impact_plotly(
    root_id: str,
    root_node: dict,
    impact_nodes: list[dict],
    relationships: list[dict],
    node_radius: float = 0.04,  # approx data-coord radius of a size-18 marker in a [-1,1] spring layout,
) -> go.Figure:
    max_depth = max((e["traversal_depth"] for e in impact_nodes), default=1)

    G = nx.DiGraph()
    G.add_node(root_id, title=root_node["title"], node_type=root_node["type"], depth=0)
    for entry in impact_nodes:
        n = entry["node"]
        G.add_node(n["id"], title=n["title"], node_type=n["type"], depth=entry["traversal_depth"])

    node_ids = set(G.nodes)
    for r in relationships:
        src, tgt = r["source_id"], r["target_id"]
        if src in node_ids and tgt in node_ids:
            G.add_edge(src, tgt, rel_type=r["rel_type"], rel_id=r.get("rel_id", ""))

    pos = nx.spring_layout(G, seed=42)

    node_x, node_y, node_text, node_hover, node_depth = [], [], [], [], []
    for nid, data in G.nodes(data=True):
        x, y = pos[nid]
        node_x.append(x)
        node_y.append(y)
        node_text.append(_truncate(data["title"], 25))
        node_hover.append(f"{data['title']} ({data['node_type']})<br>ID: {nid}<br>Depth: {data['depth']}")
        node_depth.append(data["depth"])

    node_ids_ordered = [nid for nid, _ in G.nodes(data=True)]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hovertext=node_hover,
        hoverinfo="text",
        customdata=node_ids_ordered,
        marker={
            "size": 18,
            "color": node_depth,
            "colorscale": [[0, "#1f4e79"], [1, "#a8c8e8"]],
            "cmin": 0,
            "cmax": max(max_depth, 1),
            "line": {"width": 1, "color": "#333"},
            "showscale": False,
        },
        showlegend=False,
    )

    fig = go.Figure(
        data=[node_trace],
        layout=go.Layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            hovermode="closest",
            height=400,
        ),
    )

    edge_mx, edge_my, edge_hover = [], [], []
    for src, tgt, data in G.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        edge_mx.append((x0 + x1) / 2)
        edge_my.append((y0 + y1) / 2)
        edge_hover.append(f"{data.get('rel_type', '?')}<br>Rel ID: {data.get('rel_id', 'n/a')}")

    fig.add_trace(
        go.Scatter(
            x=edge_mx,
            y=edge_my,
            mode="markers",
            marker={"size": 10, "opacity": 0},
            hovertext=edge_hover,
            hoverinfo="text",
            showlegend=False,
        )
    )

    for src, tgt, data in G.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        rel_type = data.get("rel_type", "")
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2

        # Shorten both endpoints so arrow starts and ends at node circle edges
        dx, dy = x1 - x0, y1 - y0
        length = (dx**2 + dy**2) ** 0.5 or 1
        ax0 = x0 + (dx / length) * node_radius
        ay0 = y0 + (dy / length) * node_radius
        ax1 = x1 - (dx / length) * node_radius
        ay1 = y1 - (dy / length) * node_radius

        fig.add_annotation(
            x=ax1,
            y=ay1,
            ax=ax0,
            ay=ay0,
            axref="x",
            ayref="y",
            xref="x",
            yref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=1.5,
            arrowcolor="#888",
            text="",
        )
        if rel_type:
            fig.add_annotation(
                x=mx,
                y=my,
                xref="x",
                yref="y",
                text=rel_type,
                showarrow=False,
                font={"size": 10, "color": "#ccc"},
                bgcolor="rgba(0,0,0,0.5)",
                borderpad=2,
            )

    return fig


def _truncate(s: str, max_len: int = 30) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s
