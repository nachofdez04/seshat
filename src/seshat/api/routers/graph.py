from __future__ import annotations

from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query

from seshat.api.dependencies import CurrentUser, get_app_state, require_role
from seshat.api.state import AppState
from seshat.models.api_graph import (
    BulkNodeCreate,
    BulkNodeDelete,
    BulkResult,
    ManualNodeCreate,
    ManualNodeUpdate,
    NodeFilter,
    NodeOverride,
    ResolveRequest,
    ResolveResponse,
)
from seshat.models.api_responses import (
    ImpactNode,
    ImpactResponse,
    NodeDetailResponse,
    NodeListResponse,
)
from seshat.models.enums import (
    ApprovalMethod,
    ConceptType,
    GraphDirection,
    IngestionSource,
    NodeState,
    NodeStatus,
    RelationshipType,
    UserRole,
)
from seshat.models.nodes import KBNode
from seshat.worker.manual_ingestion import NodeNotFoundError, NodePreconditionError

router = APIRouter(prefix="/graph", tags=["graph"], dependencies=[Depends(require_role(UserRole.VIEWER))])


@router.get("")
async def query_graph(
    state: Annotated[AppState, Depends(get_app_state)],
    node_type: ConceptType | None = None,
    team: str | None = None,
    project: str | None = None,
    domain: str | None = None,
    ingestion_source: IngestionSource | None = None,
    min_confidence: float | None = None,
    node_state: Annotated[NodeState | None, Query()] = None,
    node_status: Annotated[NodeStatus | None, Query()] = None,
) -> NodeListResponse:
    node_filter = NodeFilter(
        node_type=node_type,
        team=team,
        project=project,
        domain=domain,
        ingestion_source=ingestion_source,
        min_confidence=min_confidence,
        state=node_state,
        status=node_status,
    )
    nodes = await state.kb_store.query(node_filter)
    return NodeListResponse(nodes=nodes)


@router.get("/{node_id}")
async def get_node(
    node_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
) -> NodeDetailResponse:
    node = await state.kb_store.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    neighbours = await state.kb_store.get_neighbours(node_id, direction=GraphDirection.BOTH)
    active_neighbours = [n for n in neighbours if _both_current(node, n)]
    return NodeDetailResponse(node=node, neighbours=active_neighbours)


@router.get("/{node_id}/impact")
async def impact_traversal(
    node_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
    depth: Annotated[int, Query(ge=1, le=3)] = 2,
    rel_types: str | None = None,
    min_confidence: float = 0.0,
) -> ImpactResponse:
    rel_type_filter = [RelationshipType(r.strip()) for r in rel_types.split(",")] if rel_types else None

    visited: dict[str, int] = {}
    frontier = [node_id]
    for hop in range(1, depth + 1):
        next_frontier = []
        for nid in frontier:
            neighbours = await state.kb_store.get_neighbours(
                nid, rel_types=rel_type_filter, direction=GraphDirection.INBOUND
            )
            for n in neighbours:
                if str(n.id) not in visited and n.confidence >= min_confidence:
                    visited[str(n.id)] = hop
                    next_frontier.append(str(n.id))

        frontier = next_frontier

    impact_nodes: list[ImpactNode] = []
    for nid, hop in visited.items():
        n = await state.kb_store.get_node(nid)
        if n:
            impact_nodes.append(ImpactNode(node=n, traversal_depth=hop))

    return ImpactResponse(nodes=impact_nodes)


@router.post("/bulk")
async def bulk_create_nodes(
    payload: BulkNodeCreate,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> BulkResult:
    result = await state.manual_ingestion.bulk_create(payload, user.user_id)
    return result


@router.post("/nodes/resolve")
async def resolve_nodes(
    payload: ResolveRequest,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> ResolveResponse:
    nodes = []
    for node_id in payload.node_ids:
        node = await state.kb_store.get_node(str(node_id))
        if node is None:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
        nodes.append(node)

    not_approved = [str(n.id) for n in nodes if n.status != NodeStatus.APPROVED]
    if not_approved:
        raise HTTPException(status_code=422, detail=f"Nodes not in APPROVED status: {not_approved}")

    job_id = f"manual_resolve_{uuid4()}"
    relationships = await state.manual_ingestion.resolve(nodes, job_id)
    return ResolveResponse(relationships_created=len(relationships))


@router.post("", status_code=201)
async def create_node(
    payload: ManualNodeCreate,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> KBNode:
    node = await state.manual_ingestion.create(payload, user.user_id)
    return node


@router.put("/{node_id}")
async def update_node(
    node_id: str,
    payload: ManualNodeUpdate,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> KBNode:
    try:
        node = await state.manual_ingestion.update(node_id, payload, user.user_id)
    except NodeNotFoundError:
        raise HTTPException(status_code=404, detail="Node not found")
    except NodePreconditionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return node


@router.put("/{node_id}/override")
async def override_node(
    node_id: str,
    payload: NodeOverride,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> KBNode:
    minimum_method = None if user.role.is_at_least(UserRole.ADMIN) else ApprovalMethod.AUTO
    try:
        node = await state.manual_ingestion.override(node_id, payload, user.user_id, minimum_method=minimum_method)
    except NodeNotFoundError:
        raise HTTPException(status_code=404, detail="Node not found")
    except NodePreconditionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return node


@router.delete("/bulk")
async def bulk_delete_nodes(
    payload: BulkNodeDelete,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.ADMIN))],
    cascade: bool = True,
) -> BulkResult:
    result = await state.manual_ingestion.bulk_delete(payload, cascade=cascade)
    return result


@router.delete("/{node_id}", status_code=204)
async def delete_node(
    node_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.ADMIN))],
    cascade: bool = True,
) -> None:
    try:
        await state.manual_ingestion.delete(node_id, cascade=cascade)
    except NodePreconditionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


def _both_current(source: KBNode, target: KBNode) -> bool:
    return source.state == NodeState.CURRENT and target.state == NodeState.CURRENT
