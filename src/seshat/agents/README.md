# `seshat/agents`

LLM agents that power Seshat's meeting-transcript processing pipeline. There are three agent families plus a supporting verification agent.

## Folder structure

```
agents/
├── base.py                     # _BaseAgent (shared LLM retry), RetryExhaustedError
├── identification/
│   ├── base.py                 # _BaseIdentificationAgent, ConceptModel, ConceptList, AnchoredConcept
│   ├── action_item.py          # ActionItem model + ActionItemIdentificationAgent
│   ├── decision.py             # Decision model + DecisionIdentificationAgent
│   ├── open_question.py        # OpenQuestion model + OpenQuestionIdentificationAgent
│   ├── risk.py                 # Risk model + RiskIdentificationAgent
│   ├── grouping.py             # GroupingAgent (optional post-identification clustering)
│   └── registry.py             # IdentificationAgentRegistry
├── resolution/
│   ├── base.py                 # _BaseResolutionAgent, BaseSameTypeResolutionAgent,
│   │                           #   BaseCrossTypeResolutionAgent, ResolvedRelationship
│   ├── registry.py             # ResolutionRegistry (facade over both sub-registries)
│   ├── same_type/
│   │   ├── action_item.py      # ActionItemResolutionAgent
│   │   ├── decision.py         # DecisionResolutionAgent
│   │   ├── open_question.py    # OpenQuestionResolutionAgent
│   │   ├── risk.py             # RiskResolutionAgent
│   │   └── registry.py         # SameTypeResolutionRegistry
│   └── cross_type/
│       ├── action_item.py      # ActionItemCrossTypeResolutionAgent (→ Risk)
│       ├── decision.py         # DecisionCrossTypeResolutionAgent (→ Risk, OpenQuestion, ActionItem)
│       ├── open_question.py    # OpenQuestionCrossTypeResolutionAgent (→ Decision, ActionItem)
│       ├── risk.py             # RiskCrossTypeResolutionAgent (→ Decision, OpenQuestion, ActionItem)
│       └── registry.py         # CrossTypeResolutionRegistry
└── verification.py             # VerificationAgent (quote ↔ claim grounding check)
```

## `_BaseAgent` and retry contract

`_BaseAgent` (`base.py`) is the inheritance root for every LLM-calling agent. All agent families inherit from it: `_BaseIdentificationAgent`, `GroupingAgent`, `_BaseResolutionAgent`, and `VerificationAgent`.

**Retry contract:** on each transient failure the call sleeps with exponential backoff (`0.5 * 2^attempt + jitter`), then retries. After `LLMConfig.max_retries` attempts the caller-supplied `RetryExhaustedError` subclass is raised. Callers supply their own subclass so the exception hierarchy is preserved end-to-end.

### Exception hierarchy

All exhaustion errors extend `RetryExhaustedError` — catch the base class to handle any agent failure uniformly:

```
RetryExhaustedError                         (seshat.agents.base)
├── IdentificationRetryExhaustedError       (seshat.agents.identification.base)
├── GroupingRetryExhaustedError             (seshat.agents.identification.grouping)
├── ResolutionRetryExhaustedError           (seshat.agents.resolution.base)
└── VerificationRetryExhaustedError         (seshat.agents.verification)
```

## Identification agents

Each identification agent reads a meeting transcript and returns a list of `AnchoredConcept[M]` — pairs of (extracted item, `QuoteAnchor`). The anchor records the verbatim quote and its byte-offset in the transcript so downstream steps can locate the source passage.

Optionally, a concept type can enable **grouped identification**: after the raw identification pass, `GroupingAgent` clusters the items by topic and returns `list[ConceptGroup[M]]` instead.

### How they work

1. `_BaseIdentificationAgent._extract()` sends a system prompt + transcript to the LLM using LangChain's `with_structured_output`, which constrains the response to a `ConceptList[M]` schema.
2. Each extracted item's `quote` field is anchored to the transcript via `QuoteAnchor.from_transcript_quote` (fuzzy matching).
3. If `grouped_extraction` is `True`, the results are passed to `GroupingAgent`, which asks the LLM to organise the items into thematic clusters.

### Adding a new concept type

1. Create `identification/<name>.py` with a `<Name>(ConceptModel)` Pydantic model, a `<Name>List(ConceptList[<Name>]): ...` wrapper, and a `<Name>IdentificationAgent(_BaseIdentificationAgent[<Name>])` concrete class.
2. Register it in `identification/registry.py`.

## Resolution agents

Resolution agents compare **new-meeting nodes** (source) against **KB nodes** from prior meetings (targets) and emit directed `ResolvedRelationship` edges.

### Same-type resolution

`BaseSameTypeResolutionAgent` handles relationships between nodes of the same concept type (e.g. Decision → Decision). It fans out one LLM call per source node in parallel, then validates the collected relationships for:

- **Anti-symmetry** — if A supersedes B, B cannot supersede A.
- **Mutual exclusion** — a pair cannot hold both `supersedes` and `amends`; the agent keeps the weaker claim (`amends`).

Each concrete agent restricts the allowed `rel_type` values to the subset meaningful for that concept type via a `Literal` type on the `_Entry` model.

### Cross-type resolution

`BaseCrossTypeResolutionAgent` handles relationships between nodes of *different* types (e.g. Decision → Risk: `mitigates`). No anti-symmetry or mutual-exclusion checks are needed; the allowed relationships are inherently directional by definition.

Each source agent class covers one source type and potentially multiple target types. The target type is passed at construction time, which selects the correct system prompt from a `_PROMPTS` dict.

`CrossTypeResolutionRegistry` builds all 9 configured (source, target) pairs at startup using `functools.partial` to pre-bind `target_type` before passing `llm` and `config`.

### Registries

`SameTypeResolutionRegistry` and `CrossTypeResolutionRegistry` each expose a `resolve_all(source_nodes, target_nodes)` method that fans out all applicable agents concurrently with `asyncio.gather` and returns a flat list of `ResolvedRelationship`.

`ResolutionRegistry` is a thin facade that owns both sub-registries and runs them in parallel via `asyncio.gather`. This is the only entry point the orchestrator needs.

## Prompt structure

All resolution agent prompts (both same-type and cross-type) follow a standard layout:

```
[TASK DESCRIPTION]
[DEFINITIONS — what each rel_type means for this source type]
[GUARD RAILS — what doesn't qualify, boundary cases, common confusions]
[OUTPUT FORMAT]
[EXAMPLES — positive and negative]
```

Definitions appear before guard rails so the LLM grounds its understanding of each relationship before learning what to exclude. Explicit guard rails per `rel_type` reduce over-prediction on boundary cases (e.g. `amends` vs `supersedes`, `blocks` vs `mitigates`). Inline examples anchor abstract rules to concrete instances — this is especially important for cross-type relationships, where the model otherwise conflates source-type semantics with target-type semantics.

## Verification agent

`VerificationAgent` is independent of the identification/resolution families. Given a KB node's `title`, `description`, and `quote`, it asks the LLM whether the quote directly and unambiguously supports the claim. Used as a post-processing step to flag low-confidence nodes.

## Concept taxonomy and relationship ontology

### Concept types

| Type | What it captures |
|---|---|
| `Decision` | A settled choice or policy the team commits to following |
| `Risk` | A failure mode, exposure, or concern that has not yet been mitigated |
| `ActionItem` | A concrete task assigned to a person or team with expected follow-through |
| `OpenQuestion` | An unresolved question that blocks or shapes future decisions |

### Relationship matrix

Relationships are directed: *source → target*. Each cell lists the relationship types a resolution agent may assign; `—` means no agent handles that pairing.

| Source ↓ / Target → | Decision | Risk | ActionItem | OpenQuestion |
|---|---|---|---|---|
| **Decision** | `supersedes`, `amends`, `conflicts_with` | `mitigates` | `blocks` | `resolves` |
| **Risk** | `blocks` | `amends` | `blocks` | `blocks` |
| **ActionItem** | — | `mitigates` | `supersedes`, `amends`, `conflicts_with`, `blocks`, `depends_on` | — |
| **OpenQuestion** | `blocks` | — | `blocks` | `amends`, `depends_on` |

### Relationship semantics

**Same-type relationships:**

- **`supersedes`** — the source replaces the target entirely; the target is no longer active. Used for `Decision` and `ActionItem` only.
- **`amends`** — the source narrows, qualifies, or refines the target without replacing it; the target remains active. Always directed from the more specific to the more general. Used for all four types.
- **`conflicts_with`** — both nodes are currently active and mutually incompatible; following one makes the other impossible to follow. Only valid when the target is active; never assigned toward a deferred, rejected, or superseded node. Used for `Decision` and `ActionItem`.
- **`blocks`** (same-type, `ActionItem` only) — the source task must complete before the target can start; the target is still needed.
- **`depends_on`** — the source cannot proceed without the target being completed or answered first. Used for `ActionItem` and `OpenQuestion`.

**Cross-type relationships:**

- **`mitigates`** — the source (`Decision` or `ActionItem`) directly reduces or controls the target risk's failure mode, likelihood, or blast radius. Domain-level proximity or incidental benefit does not qualify; the source must mechanistically address the risk's specific failure mode.
- **`resolves`** — the source `Decision` provides a direct, complete, and final answer to the target `OpenQuestion`. Partial answers, phase-scoped choices, or decisions that assume an answer do not qualify.
- **`blocks`** (cross-type) — the source creates a direct obstacle that prevents the target from being safely or meaningfully executed as stated:
  - `Decision → ActionItem`: the decision imposes a freeze, prohibition, or incompatible constraint on the action item.
  - `Risk → Decision | OpenQuestion | ActionItem`: the unresolved risk makes it unsafe, illegal, or operationally impossible to execute or answer the target.
  - `OpenQuestion → Decision | ActionItem`: the unanswered question withholds a required parameter, constraint, or prerequisite for the target.

  Contextual or domain-level relevance does not qualify; the source must be a concrete obstacle to the target proceeding as stated.

### Known limitations

**Ownerless action items are not captured.** `ActionItem` enforces `assignee: str` (non-nullable) and the identification prompt suppresses items without an identifiable owner. Tasks that emerged without a named person or role (e.g. "someone from the platform team needs to handle this") are silently dropped. This is intentional — capturing ownerless tasks at reduced confidence is deferred until there is evidence of need from real transcripts.

### Design rationale

**Why these four types?** They map directly to what technical meetings produce: a choice made (`Decision`), a concern surfaced (`Risk`), a task assigned (`ActionItem`), and a question left open (`OpenQuestion`). The model captures the four kinds of output with durable cross-meeting relevance that teams most often lose track of — not everything discussed.

**What the taxonomy deliberately omits:**
- *Discussion and debate history.* Within a single meeting, only the final settled outcome survives. Earlier reversed positions are discarded during within-meeting deduplication. `SUPERSEDES` is reserved for cross-meeting evolution; it is never created within a single job.
- *Meeting content that doesn't generate structured output.* Status updates, announcements, and exploratory discussion are not identified. The KB represents conclusions, not the path to them.

**Why these relationships?** The schema is narrow by design. Each relationship type is defined only for (source type, target type) pairings where it has a clear, mechanistic meaning:
- `supersedes` and `amends` track evolution within a concept type over time.
- `conflicts_with` surfaces active contradictions for human resolution — it does not automatically change either node's state.
- `blocks` and `depends_on` capture execution dependencies and cross-type cases where an unresolved concern genuinely gates another node.
- `mitigates` and `resolves` are the two "closing" relationships: a `Decision` or `ActionItem` closing out a `Risk`, and a `Decision` settling an `OpenQuestion`.

**`conflicts_with` does not change node state.** Both conflicting nodes remain `CURRENT`; the conflict is surfaced at review time. The system does not decide which of two conflicting decisions is correct.

Same-type anti-symmetry and mutual-exclusion rules are enforced post-resolution (see *Same-type resolution* above).

## Type-system notes

### `Generic[M]` on identification types (`identification/base.py`)

```python
M = TypeVar("M", bound=ConceptModel)

class ConceptList(BaseModel, Generic[M]):
    items: list[M]

class AnchoredConcept(BaseModel, Generic[M]):
    item: M
    quote_anchor: QuoteAnchor | None

class _BaseIdentificationAgent(Generic[M]): ...
```

`M` is the concrete model type (`Decision`, `Risk`, etc.). Making the base classes generic lets pyright track the concrete type end-to-end — `extract()` returns `list[AnchoredConcept[Decision]]` for `DecisionIdentificationAgent`, not `list[AnchoredConcept[ConceptModel]]`.

### `Generic[E]` on `_ResultBase` (`resolution/base.py`)

```python
E = TypeVar("E", bound=_EntryBase)

class _ResultBase(BaseModel, Generic[E]):
    entries: list[E] = Field(default_factory=list)
```

`_EntryBase` subclasses narrow `rel_type` to a `Literal` of the allowed relationship types for that concept. Without the generic, each `_XxxResult` subclass would have to redeclare `entries: list[_XxxEntry]` as a field override — which pyright rejects because `list` is invariant. Making `_ResultBase` generic lets subclasses inherit the correctly-typed field by parameterising the base:

```python
class _DecisionResult(_ResultBase[_DecisionEntry]): ...
```

### `Generic[E]` on the resolution agent hierarchy (`resolution/base.py`)

```python
class _BaseResolutionAgent(Generic[E]):
    @property
    @abstractmethod
    def _result_model(self) -> type[_ResultBase[E]]: ...

class BaseSameTypeResolutionAgent(_BaseResolutionAgent[E]): ...
class BaseCrossTypeResolutionAgent(_BaseResolutionAgent[E]): ...

class DecisionResolutionAgent(BaseSameTypeResolutionAgent[_DecisionEntry]): ...
```

`E` propagates from the abstract base through the intermediate bases to the concrete class, which fixes it to the specific entry type. `_result_model` is fully typed end-to-end with no `Any` escape hatch.

### `# type: ignore[override]` on `rel_type` (`resolution/*/decision.py` etc.)

```python
class _DecisionEntry(_EntryBase):
    rel_type: Literal[RelationshipType.SUPERSEDES, ...] | None  # type: ignore[override]
```

`_EntryBase.rel_type` is `str | None`. The subclass narrows it to a `Literal`. This is a valid Pydantic narrowing at runtime, but pyright considers it an incompatible override because `Literal[...]` is not a subtype of `str` in the type-override sense. The ignore is intentional and localised.
