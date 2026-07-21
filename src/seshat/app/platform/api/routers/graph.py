from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from seshat.app.platform.api.dependencies import CurrentUser, get_app_state, require_role
from seshat.app.platform.api.state import AppState
from seshat.app.services.graph import (
    NodeNotFoundError,
    NodePreconditionError,
    RelationshipConflictError,
    RelationshipNotFoundError,
)
from seshat.core.models.api_graph import (
    BulkNodeCreate,
    BulkNodeDelete,
    BulkResult,
    ManualNodeCreate,
    ManualNodeUpdate,
    NodeFilter,
    NodeOverride,
    RelationshipCreateRequest,
    ResolveRequest,
    ResolveResponse,
)
from seshat.core.models.api_responses import (
    ImpactResponse,
    NodeDetailResponse,
    NodeListResponse,
    NodeSearchResponse,
    RelationshipListResponse,
)
from seshat.core.models.enums import ApprovalMethod, GraphDirection, RelationshipType, SearchMode, UserRole
from seshat.core.models.nodes import KBNode, KBRelationship

# Parent router carries the /graph prefix and the baseline VIEWER auth gate.
# Sub-routers group by resource; fixed prefixes avoid catch-all routing conflicts.
router = APIRouter(prefix="/graph", tags=["graph"], dependencies=[Depends(require_role(UserRole.VIEWER))])

_nodes_router = APIRouter(prefix="/nodes", tags=["graph nodes"])
_relations_router = APIRouter(prefix="/relationships", tags=["graph relationships"])

# Sub-routers are included first so their fixed prefixes take priority over the
# /{node_id} catch-all registered below.
router.include_router(_nodes_router)
router.include_router(_relations_router)


# -- KB and VS browse ---------------------------------------------------------


@router.get(
    "",
    summary="Query knowledge graph nodes",
    responses={
        200: {"description": "List of matching nodes (may be empty)"},
        401: {"description": "Missing or invalid API key"},
    },
)
async def query_graph(
    state: Annotated[AppState, Depends(get_app_state)],
    node_filter: Annotated[NodeFilter, Depends()],
) -> NodeListResponse:
    nodes = await state.graph_service.query(node_filter)
    return NodeListResponse(nodes=nodes)


@router.get(
    "/search",
    summary="Hybrid semantic search over KB nodes",
    responses={
        200: {"description": "Matching nodes with neighbours, ordered by relevance"},
        401: {"description": "Missing or invalid API key"},
    },
)
async def search_graph(
    state: Annotated[AppState, Depends(get_app_state)],
    q: str,
    node_filter: Annotated[NodeFilter, Depends()],
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
    search_mode: SearchMode = SearchMode.SEMANTIC,
    score_threshold: Annotated[float | None, Query(ge=0, le=1)] = None,
) -> NodeSearchResponse:
    results = await state.graph_service.search(
        query=q, limit=limit, node_filter=node_filter, mode=search_mode, score_threshold=score_threshold
    )
    return NodeSearchResponse(results=results)


@router.get(
    "/{node_id}",
    summary="Fetch a single KB node by ID",
    responses={
        200: {"description": "The node"},
        401: {"description": "Missing or invalid API key"},
        404: {"description": "Node not found"},
    },
)
async def get_node(
    node_id: UUID,
    state: Annotated[AppState, Depends(get_app_state)],
) -> KBNode:
    try:
        return await state.graph_service.get_node(node_id)
    except NodeNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")


@router.get(
    "/{node_id}/neighbours",
    summary="List depth-1 neighbours of a node (both directions, active only)",
    responses={
        200: {"description": "Directly connected active nodes"},
        401: {"description": "Missing or invalid API key"},
        404: {"description": "Node not found"},
    },
)
async def get_node_neighbours(
    node_id: UUID,
    state: Annotated[AppState, Depends(get_app_state)],
) -> list[KBNode]:
    try:
        return await state.graph_service.get_node_neighbours(node_id)
    except NodeNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")


@router.get(
    "/{node_id}/detail",
    summary="Fetch a node together with its depth-1 neighbours",
    responses={
        200: {"description": "Node and directly connected active nodes"},
        401: {"description": "Missing or invalid API key"},
        404: {"description": "Node not found"},
    },
)
async def get_node_detail(
    node_id: UUID,
    state: Annotated[AppState, Depends(get_app_state)],
) -> NodeDetailResponse:
    try:
        return await state.graph_service.get_node_detail(node_id)
    except NodeNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")


@router.get(
    "/{node_id}/impact",
    summary="Multi-hop impact traversal — nodes connected to this one in the chosen direction",
    responses={
        200: {"description": "Connected nodes with their traversal depth"},
        401: {"description": "Missing or invalid API key"},
        422: {"description": "depth out of allowed range [1, 3]"},
    },
)
async def impact_traversal(
    node_id: UUID,
    state: Annotated[AppState, Depends(get_app_state)],
    depth: Annotated[int, Query(ge=1, le=3)] = 2,
    rel_types: str | None = None,
    min_confidence: float = 0.0,
    direction: GraphDirection = GraphDirection.OUTBOUND,
) -> ImpactResponse:
    rel_type_filter: list[RelationshipType] | None
    if rel_types:
        try:
            rel_type_filter = [RelationshipType(r.strip()) for r in rel_types.split(",")]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    else:
        rel_type_filter = None
    return await state.graph_service.traverse_impact(node_id, depth, rel_type_filter, min_confidence, direction)


# -- Node manual actions ------------------------------------------------------


@_nodes_router.post(
    "/bulk",
    summary="Bulk create nodes",
    responses={
        200: {"description": "Succeeded and failed node IDs"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Operator role required"},
    },
)
async def bulk_create_nodes(
    payload: BulkNodeCreate,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> BulkResult:
    try:
        return await state.graph_service.bulk_create(payload, user.user_id)
    except NodePreconditionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@_nodes_router.post(
    "/resolve",
    summary="Trigger resolution for a set of approved nodes",
    responses={
        200: {"description": "Number of relationships created"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Operator role required"},
        404: {"description": "One or more node IDs not found"},
        422: {"description": "One or more nodes not in APPROVED status"},
    },
)
async def resolve_nodes(
    payload: ResolveRequest,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> ResolveResponse:
    try:
        relationships = await state.graph_service.resolve_by_ids(payload.node_ids)
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except NodePreconditionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    return ResolveResponse(relationships_created=relationships)


@_nodes_router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Manually create a node",
    responses={
        201: {"description": "Node created"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Operator role required"},
    },
)
async def create_node(
    payload: ManualNodeCreate,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> KBNode:
    return await state.graph_service.create(payload, user.user_id)


@_nodes_router.put(
    "/{node_id}",
    summary="Alter a manually-created node",
    responses={
        200: {"description": "Updated node"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Operator role required"},
        404: {"description": "Node not found"},
        409: {"description": "Node not eligible for update (e.g. not manually created)"},
    },
)
async def update_node(
    node_id: UUID,
    payload: ManualNodeUpdate,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> KBNode:
    try:
        return await state.graph_service.update(node_id, payload, user.user_id)
    except NodeNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")
    except NodePreconditionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@_nodes_router.put(
    "/{node_id}/override",
    summary="Alter any node with correction metadata, role-gated",
    responses={
        200: {"description": "New node version created"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Operator role required"},
        404: {"description": "Node not found"},
        409: {"description": "Node not eligible for override"},
    },
)
async def override_node(
    node_id: UUID,
    payload: NodeOverride,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> KBNode:
    minimum_method = None if user.role.is_at_least(UserRole.ADMIN) else ApprovalMethod.AUTO
    try:
        return await state.graph_service.override(node_id, payload, user.user_id, minimum_method=minimum_method)
    except NodeNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Node not found")
    except NodePreconditionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


@_nodes_router.delete(
    "/bulk",
    summary="Bulk delete nodes",
    responses={
        200: {"description": "Succeeded and failed node IDs"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Admin role required"},
    },
)
async def bulk_delete_nodes(
    payload: BulkNodeDelete,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.ADMIN))],
    cascade: bool = True,
) -> BulkResult:
    try:
        return await state.graph_service.bulk_delete(payload, cascade=cascade)
    except NodePreconditionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@_nodes_router.delete(
    "/{node_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a node",
    responses={
        204: {"description": "Node deleted"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Admin role required"},
        409: {"description": "Cannot delete node with inbound relationships (use cascade=true)"},
    },
)
async def delete_node(
    node_id: UUID,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.ADMIN))],
    cascade: bool = True,
) -> None:
    try:
        await state.graph_service.delete(node_id, cascade=cascade)
    except NodePreconditionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))


# -- Relationship manual actions ----------------------------------------------


@_relations_router.get(
    "",
    summary="List relationships, optionally filtered by node and/or type",
    responses={
        200: {"description": "Matching relationships (may be empty)"},
        401: {"description": "Missing or invalid API key"},
    },
)
async def list_relationships(
    state: Annotated[AppState, Depends(get_app_state)],
    node_id: UUID | None = None,
    rel_type: RelationshipType | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> RelationshipListResponse:
    rels = await state.graph_service.list_relationships(node_id=node_id, rel_type=rel_type, limit=limit)
    return RelationshipListResponse(relationships=rels)


@_relations_router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Manually create a relationship between two existing nodes",
    responses={
        201: {"description": "Relationship created"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Operator role required"},
        404: {"description": "Source or target node not found"},
        409: {"description": "Identical relationship already exists"},
    },
)
async def create_relationship(
    payload: RelationshipCreateRequest,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> KBRelationship:
    try:
        return await state.graph_service.create_relationship(payload.source_id, payload.target_id, payload.rel_type)
    except NodeNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RelationshipConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@_relations_router.delete(
    "/{rel_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a relationship by surrogate ID",
    responses={
        204: {"description": "Relationship deleted"},
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Admin role required"},
        404: {"description": "Relationship not found"},
    },
)
async def delete_relationship(
    rel_id: UUID,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.ADMIN))],
) -> None:
    try:
        await state.graph_service.delete_relationship(rel_id)
    except RelationshipNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
