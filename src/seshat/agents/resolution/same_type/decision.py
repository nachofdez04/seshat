from typing import Literal

from seshat.agents.resolution.base import BaseSameTypeResolutionAgent, _ResultBase, _SameTypeEntry
from seshat.models.enums import RelationshipType

_DECISION_RELATION_TYPES = Literal[
    RelationshipType.SUPERSEDES, RelationshipType.AMENDS, RelationshipType.CONFLICTS_WITH
]


class _DecisionEntry(_SameTypeEntry):
    rel_type: _DECISION_RELATION_TYPES | None  # type: ignore[override]
    alt_rel_type: _DECISION_RELATION_TYPES | None = None  # type: ignore[override]


class _DecisionResult(_ResultBase[_DecisionEntry]): ...


_DECISION_PROMPT = """\
You are a decision relationship resolution agent.

## Relation definitions

### conflicts_with
Both decisions are currently active, address the same concern, and are mutually incompatible — following one makes it impossible to follow the other.
- Example: "Set token lifetime to 15 minutes" conflicts_with "Set token lifetime to 24 hours".

### supersedes
The source permanently replaces the target in the same concern domain, rendering it no longer active.
- Example: "Use PostgreSQL for all storage" supersedes "Use SQLite for all storage".

### amends
The source modifies the target (qualifies, narrows, extends, or adds an exception) without replacing it. Both address the same concern; the target remains broadly active.
- Example: "Enforce two-approver sign-off for production deployments only" amends "All deployments require two approvals".

## Task
You receive a source decision (current meeting) and a list of target decisions (prior meeting KB).
For each target, output one rel_type. Every target MUST appear in the output.
Relationships are directed: source → target only.

## Over-extraction guards

### Same concern domain (hard stop)
Before assigning any label, confirm both decisions govern the same policy area, architectural layer, or system component. Ask: "Are they in the same concern domain?" If no → null; do not proceed.
- Example: "Require TLS for token endpoints" is NOT any relation for "Use session tokens for authentication" — transport security vs auth mechanism → null.

### Not conflicts_with
- The target is inactive (deferred, rejected, or on hold) — conflicts_with requires both decisions to be currently active.
  Counter-example: "Use PostgreSQL for all storage" is NOT conflicts_with "Use SQLite for all storage (deprecated)" — the old decision is inactive → supersedes.

### Not supersedes
- The source is a temporary restriction (freeze, hold, moratorium) — it does not permanently replace the policy it restricts → null.
- The source narrows, qualifies, or adds an exception to the target without replacing it → amends instead.
- The source makes the target indirectly stale as a side effect of replacing something the target parameterises (e.g., source replaces a mechanism; target configured that mechanism). Indirect staleness does not qualify — the source must directly address the same policy as the target → null.
- When ambiguous between supersedes and amends, prefer amends.
  Counter-example: "All production deployments must use immutable artefact promotion" is NOT supersedes for "Database schema changes must use in-place upgrades" — both are active in the same deployment domain; source does not render the target defunct → conflicts_with.

### Not amends
- amends is directed from the more specific to the more general. If both qualify each other equally, prefer null.
  Counter-example: "Require TLS for token endpoints" is NOT amends for "Use session tokens for authentication" — different concern domains (transport vs auth mechanism) → null.
- The source is a temporary freeze, moratorium, or execution gate — it controls *when* the target's policy may be enacted, not *what the policy says*. A governance gate suspends execution; it does not qualify or narrow the policy itself → null.

## Selection
Once the same concern domain guard passes, select the first label that applies:
1. **conflicts_with** — if both decisions are currently active and co-enforcing them would produce a logical contradiction. Ask: "Would enforcing both simultaneously lead to a contradiction?" For policies with different scope, test per entity: "Is there any single entity (service, user, team, resource) covered by both policies where following one makes the other impossible?" If yes → conflicts_with. Broader scope does not make a relationship amends; scope expansion that forces some entities into contradicting their prior policy is a conflict.
2. **supersedes** — if the source permanently replaces the target, rendering it no longer active. Ask: "Is the target's policy now defunct?" If yes → supersedes.
3. **amends** — if the source qualifies, narrows, extends, or adds an exception to the target, while the target remains broadly active. Ask: "Does the target remain a valid standing policy after the source exists?" If yes → amends.
4. Otherwise → null.

## Boundary examples
- conflicts_with vs supersedes:
  - "All services retry at most 3 times" → conflicts_with "All services retry at most 10 times" — both active blanket policies, contradictory values; neither is inactive.
  - "All services retry at most 3 times" → supersedes "Services may retry indefinitely" — source permanently closes the old open-ended approach.

## Ambiguity signal
If, after applying the selection rules above, you are genuinely uncertain between two specific
relationship types (not between a type and null), set alt_rel_type to the runner-up.
alt_rel_type must be one of the same valid types as rel_type, and must differ from rel_type.
Leave alt_rel_type null for clear-cut cases, null assignments, and when uncertain between a type and null.
"""


class DecisionResolutionAgent(BaseSameTypeResolutionAgent[_DecisionEntry]):
    @property
    def _result_model(self) -> type[_DecisionResult]:
        return _DecisionResult

    @property
    def _system_prompt(self) -> str:
        return _DECISION_PROMPT
