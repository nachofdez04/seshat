# Manual KB Endpoints — Design Spec

**Date:** 2026-06-27
**Status:** Draft
**Supersedes:** Non-goal items at seshat-design.md lines 33–34 (standalone `POST /graph` and `PATCH /graph/{node_id}` deferred to v2 — this spec brings them into v1 with scoped constraints).

---

## Motivation

The pipeline covers the happy path: audio/text in → LLM extraction → human review → KB written. Several gaps remain:

1. **Operators cannot add nodes outside of a job.** A known decision made in a side-channel (email, whiteboard) has no ingestion path.
2. **Operators cannot correct a manually-created node.** Once created, it is frozen even though the operator — not an LLM — is the authoritative source.
3. **Admins cannot override incorrect reviewer decisions.** Humans make mistakes; the role hierarchy should allow the highest role to fix them, with a full audit trail.
4. **Nodes cannot be deleted.** Stale or erroneous nodes accumulate with no removal path.
5. **No bulk operations.** Creating or deleting many nodes requires repeated single calls.

---

## Scope

Five new endpoints under `/v1/graph`, plus bulk counterparts for create and delete:

| Verb | Path | Role | Status | Description |
|---|---|---|---|---|
| `POST` | `/v1/graph` | OPERATOR | 201 | Create a node directly in the KB |
| `POST` | `/v1/graph/bulk` | OPERATOR | 200 | Create multiple nodes (deferred — bulk model TBD) |
| `PUT` | `/v1/graph/{node_id}` | OPERATOR | 200 | Replace a manually-created node's content |
| `PUT` | `/v1/graph/{node_id}/override` | OPERATOR | 200 | Override an auto-approved node (OPERATOR) or any node (ADMIN) |
| `DELETE` | `/v1/graph/{node_id}` | ADMIN | 204 | Delete a node and cascade its relationships |
| `DELETE` | `/v1/graph/bulk` | ADMIN | 200 | Delete multiple nodes (deferred — bulk model TBD) |

The existing `GET` endpoints are unchanged.

---

## Enum additions

Both `IngestionSource` and `ApprovalMethod` in `src/seshat/models/enums.py` are missing `MANUAL`. Add:

```python
class IngestionSource(StrEnum):
    JOB = auto()
    INIT = auto()
    MANUAL = auto()   # created directly by an operator via POST /graph

class ApprovalMethod(StrEnum):
    INDIVIDUAL = auto()
    BULK = auto()
    AUTO = auto()
    THRESHOLD = auto()
    MANUAL = auto()   # node was created manually, not reviewed from a pipeline result
```

---

## `NodeMetadata` addition

Add `correction_reason` to `NodeMetadata` in `src/seshat/models/nodes.py`:

```python
correction_reason: str | None = Field(
    default=None,
    description="Human-provided justification for correcting or overriding this node."
)
```

Set alongside `corrected_by` / `corrected_at` on any edit or override.

---

## API models

Add to `src/seshat/models/api.py`. All extend `BaseModel` (not `SeshatModel`) — request body models are not domain facts and must not be frozen.

```python
class RelationshipInput(BaseModel):
    target_id: str                  # UUID of the existing target node
    rel_type: RelationshipType

class ManualNodeCreate(BaseModel):
    type: ConceptType
    title: str
    description: str
    source_quote: str | None = None
    blob_key: str | None = None     # co-required with source_quote
    participants: list[str] | None = None
    team: str | None = None
    project: str | None = None
    domain: str | None = None
    meeting_date: date | None = None
    concept_fields: dict[str, Any] | None = None
    relationships: list[RelationshipInput] | None = None

    # model_validator: source_quote and blob_key are co-required —
    # providing one without the other is a 422 validation error

class ManualNodeUpdate(BaseModel):
    title: str
    description: str
    participants: list[str] | None = None
    team: str | None = None
    project: str | None = None
    domain: str | None = None
    meeting_date: date | None = None
    concept_fields: dict[str, Any] | None = None
    relationships: list[RelationshipInput] | None = None
    # type, confidence, quote_anchors, status, state, ingestion_source are intentionally
    # excluded — immutable even for MANUAL nodes

class NodeOverride(BaseModel):
    title: str
    description: str
    reason: str                     # required — justification stored in correction_reason
    participants: list[str] | None = None
    team: str | None = None
    project: str | None = None
    domain: str | None = None
    meeting_date: date | None = None
    concept_fields: dict[str, Any] | None = None
    relationships: list[RelationshipInput] | None = None

class BulkNodeCreate(BaseModel):
    nodes: list[ManualNodeCreate]
    on_error: Literal["stop", "continue"] = "stop"

class BulkNodeDelete(BaseModel):
    node_ids: list[str]
    on_error: Literal["stop", "continue"] = "stop"

class BulkResult(BaseModel):
    succeeded: list[str]            # node_ids (or created node UUIDs) that succeeded
    failed: list[dict]              # list of {node_id, error} for failed items
```

---

## Node construction rules

### POST /graph — create

- `id`: fresh `uuid4()`
- `schema_version`: `"1.0"`
- `type`: from payload
- `title`, `description`: from payload
- `confidence`: `1.0` (operator is the source of truth)
- `quote_anchors`: computed from `source_quote` + `blob_key` if both provided; falls back to `[]` with a `logging.warning` if the quote cannot be located in the blob; `[]` if neither is provided. If `blob_key` is provided but the blob is unreachable → 503.
- `status`: `APPROVED`
- `state`: `CURRENT`
- `metadata.ingestion_source`: `MANUAL`
- `metadata.approval_method`: `MANUAL`
- `metadata.approved_by`: requesting user's `user_id`
- `metadata.approved_at`: UTC now
- `metadata.job_id`: `"manual"` — deliberate non-UUID sentinel; no `ops.jobs` row exists. No code must attempt to dereference this as a real job ID.
- `metadata.meeting_date`, `participants`, `team`, `project`, `domain`, `concept_fields`: from payload

Relationships in `payload.relationships` are written after the node inside the same transaction. Each `KBRelationship` row uses `job_id="manual"`. Unknown `target_id` values → 422 before the transaction opens.

### POST /graph/bulk — bulk create

Iterates `ManualIngestionService.create` for each item in `nodes`. Behaviour controlled by `on_error`:
- `"stop"` (default): abort on first failure, no partial writes (each item runs in its own transaction; stop iterating on error)
- `"continue"`: attempt all items, collect successes and failures, return `BulkResult`

Returns `BulkResult` (200) in all cases. `succeeded` contains the UUIDs of created nodes.

### PUT /graph/{node_id} — update (MANUAL nodes only)

1. Fetch existing node via `get_node(node_id)`. If `None` → 404.
2. **Precondition:** `node.metadata.ingestion_source == MANUAL`. Any other value → 409 `{"detail": "Only manually-created nodes can be edited — use PUT /{node_id}/override for pipeline nodes"}`.
3. Build updated `KBNode` by reading the existing node and applying only the mutable fields from the payload. All other fields (`ingestion_source`, `approval_method`, `approved_by`, `approved_at`, `job_id`, `confidence_breakdown`, `pending_reason`) are preserved verbatim.
4. Set `metadata.corrected_by` = requesting user's `user_id`, `metadata.corrected_at` = UTC now. `metadata.correction_reason` is not set (no `reason` field on `ManualNodeUpdate` — the operator is already the source of truth).

Fields that may be updated: `title`, `description`, `participants`, `team`, `project`, `domain`, `meeting_date`, `concept_fields`.

Fields immutable even for MANUAL nodes: `type`, `confidence`, `quote_anchors`, `status`, `state`, `ingestion_source`. `quote_anchors` is not updatable — if the source quote is wrong, delete and re-create.

Relationships: if `payload.relationships` is non-null, delete all outbound `kb_relationships` where `source_id = node_id`, then insert the new set inside the same transaction. If `null`, existing relationships are untouched.

Re-embedding: `vector_store.upsert(node.id, f"{node.title} {node.description}", metadata)` inside the transaction before commit (same formula as `WritingStage`).

Returns the full updated `KBNode` (200).

### PUT /graph/{node_id}/override — override (pipeline nodes)

Role-conditional precondition:

| Requesting role | Allowed nodes |
|---|---|
| OPERATOR | `approval_method=AUTO` only |
| ADMIN | any node |

Steps:
1. Fetch existing node. If `None` → 404.
2. Check precondition (table above). Failure → 409 `{"detail": "Insufficient role to override this node"}`.
3. Build updated `KBNode`: same merge strategy as PUT (preserve all non-mutable fields), plus set `metadata.correction_reason = payload.reason`.
4. Set `metadata.corrected_by`, `metadata.corrected_at`.

Relationships: same replace-if-non-null logic as PUT.

Re-embedding: same as PUT.

Returns the full updated `KBNode` (200).

### DELETE /graph/{node_id}?cascade=true — delete

Query parameter: `cascade: bool = True`

**`cascade=true` (default):** delete all relationships where `source_id = node_id OR target_id = node_id`, then delete the node. Always succeeds regardless of inbound edges.

**`cascade=false` (safe mode):** check `count_inbound_relationships(node_id)` first. If > 0 → 409 `{"detail": "Node is referenced as a target by {n} relationships — delete them first or use cascade=true"}`. Otherwise delete only outbound relationships (`source_id = node_id`) then the node.

Sequence inside a single `kb_store.transaction()`:
1. `delete_relationships_for_node(node_id, cascade=cascade)` — see above
2. `delete_node(node_id)` — DELETE WHERE `node_id = $1`
3. `vector_store.delete(node_id)` — `adelete(ids=[node_id])`; missing ID is a no-op

Return 204 unconditionally — idempotent, no 404 on missing node. The inbound-edge check for safe mode is done **before** opening the transaction.

### DELETE /graph/bulk — bulk delete

Iterates `ManualIngestionService.delete` for each item in `node_ids`. Same `on_error` semantics as bulk create. Returns `BulkResult` (200); `succeeded` contains the node_ids that were deleted (or were already absent).

---

## `PostgresKBStore` additions

Three new methods on `PostgresKBStore` in `src/seshat/knowledge_store/pg_store.py`:

```python
async def update_node(self, node: KBNode, *, conn: _Conn | None = None) -> None:
    """UPDATE title, description, metadata (full JSON), schema_version.
    Does NOT touch quote_anchors (immutable top-level column, separate from metadata JSON).
    Raises KeyError if node_id not found (defensive guard; service layer checks existence first)."""

async def delete_node(self, node_id: str, *, conn: _Conn | None = None) -> None:
    """DELETE the node row. Caller must call delete_relationships_for_node first
    within the same transaction to satisfy FK constraints."""

async def count_inbound_relationships(self, node_id: str) -> int:
    """Return count of kb_relationships rows where target_id = node_id. Used for safe-delete check."""

async def delete_relationships_for_node(self, node_id: str, *, cascade: bool = True, conn: _Conn | None = None) -> None:
    """cascade=True: DELETE WHERE source_id=$1 OR target_id=$1.
    cascade=False: DELETE WHERE source_id=$1 only (safe mode — caller ensures no inbound edges)."""
```

`update_node` SQL columns: `title`, `description`, `metadata`, `schema_version`. Explicitly **excludes** `quote_anchors`.

Both `source_id` and `target_id` in `kb_relationships` have FK constraints pointing to `kb_nodes.node_id` (confirmed in migration `001`). Cascade-delete of both directions is therefore required to avoid FK violations when cascade=True.

---

## `ManualIngestionService`

New file: `src/seshat/worker/manual_ingestion.py` — placed in `worker/` to match `WritingStage`.

```python
class ManualIngestionService:
    def __init__(self, kb_store: PostgresKBStore, vector_store: PGVectorStore) -> None: ...

    async def create(self, payload: ManualNodeCreate, user_id: str) -> KBNode:
        """Build KBNode, write to kb_store + vector_store inside a transaction.
        Writes KBRelationship rows (job_id='manual') after the node."""

    async def update(self, node_id: str, payload: ManualNodeUpdate, user_id: str) -> KBNode:
        """get_node → 404 if missing → 409 if not MANUAL → merge fields →
        update_node + vector upsert inside transaction. Returns updated KBNode."""

    async def override(self, node_id: str, payload: NodeOverride, user_id: str, minimum_method: ApprovalMethod | None) -> KBNode:
        """get_node → 404 if missing → check precondition (minimum_method=AUTO for OPERATOR,
        None for ADMIN meaning any node) → merge fields → update_node + vector upsert.
        Sets correction_reason from payload.reason."""

    async def delete(self, node_id: str) -> None:
        """delete_relationships_for_node → delete_node → vector delete, all in one transaction."""
```

The `override` method receives the effective minimum `ApprovalMethod` from the router (derived from the requesting user's role), keeping the role logic in the router layer. Passing `minimum_method=None` means no restriction (ADMIN path).

Vector store calls happen inside the `kb_store.transaction()` block before commit. Text for upsert: `f"{node.title} {node.description}"`.

---

## `AppState` and wiring

Add `manual_ingestion: ManualIngestionService` to `AppState` in `src/seshat/api/state.py` behind a `TYPE_CHECKING` guard (matching the existing pattern for all other fields).

Add `vector_store` and `manual_ingestion` fields to `WorkerContext` in `src/seshat/worker/bootstrap.py`. Construct `ManualIngestionService(kb_store, vector_store)` inside `build_worker_context`. Wire into `AppState` from `ctx.manual_ingestion` in `app.py`'s `_lifespan`.

---

## Graph router additions

In `src/seshat/api/routers/graph.py`:

```python
@router.post("", status_code=201)
async def create_node(
    payload: ManualNodeCreate,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> KBNode: ...

@router.post("/bulk")
async def bulk_create_nodes(
    payload: BulkNodeCreate,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> BulkResult: ...

@router.put("/{node_id}")
async def update_node(
    node_id: str,
    payload: ManualNodeUpdate,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> KBNode: ...

@router.put("/{node_id}/override")
async def override_node(
    node_id: str,
    payload: NodeOverride,
    state: Annotated[AppState, Depends(get_app_state)],
    user: Annotated[CurrentUser, Depends(require_role(UserRole.OPERATOR))],
) -> KBNode:
    # Derive minimum_method from role: ADMIN → None (any node), OPERATOR → AUTO only
    minimum_method = None if user.role.is_at_least(UserRole.ADMIN) else ApprovalMethod.AUTO
    ...

@router.delete("/{node_id}", status_code=204)
async def delete_node(
    node_id: str,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.ADMIN))],
) -> None: ...

@router.delete("/bulk")
async def bulk_delete_nodes(
    payload: BulkNodeDelete,
    state: Annotated[AppState, Depends(get_app_state)],
    _user: Annotated[CurrentUser, Depends(require_role(UserRole.ADMIN))],
) -> BulkResult: ...
```

**Route ordering note:** FastAPI matches routes top-to-bottom. `POST /bulk` and `DELETE /bulk` must be registered **before** `POST ""` / `DELETE /{node_id}` respectively to avoid `/bulk` being captured as a `{node_id}` path parameter. The router definitions above already reflect correct ordering.

---

## Main spec update

Replace lines 33–34 in `seshat-design.md` non-goals section with:

```
- Post-approval node editing and standalone node creation: implemented in v1 with scoped constraints
  (MANUAL ingestion source only for standard edits; override endpoint for pipeline nodes, role-gated;
  see docs/superpowers/specs/2026-06-27-manual-kb-endpoints.md).
  Review-time creation via `ApproveRequest.created_nodes` remains in scope and unchanged.
```

---

## What is NOT in scope

- Resolution pass for manually-created nodes (no LLM calls; relationships are operator-defined)
- `ApproveRequest.created_nodes` (existing review-time creation path; unchanged)
- Soft delete / tombstoning (hard delete only for MVP)
- `PATCH` (partial update) — `PUT` replaces all mutable fields wholesale; partial update deferred
