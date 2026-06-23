from typing import Literal

from seshat.agents.resolution.base import BaseSameTypeResolutionAgent, _ResultBase, _SameTypeEntry
from seshat.models.enums import RelationshipType

_ACTION_ITEM_RELATION_TYPES = Literal[
    RelationshipType.SUPERSEDES,
    RelationshipType.AMENDS,
    RelationshipType.CONFLICTS_WITH,
    RelationshipType.BLOCKS,
    RelationshipType.DEPENDS_ON,
]


class _ActionItemEntry(_SameTypeEntry):
    rel_type: _ACTION_ITEM_RELATION_TYPES | None  # type: ignore[override]
    alt_rel_type: _ACTION_ITEM_RELATION_TYPES | None = None  # type: ignore[override]


class _ActionItemResult(_ResultBase[_ActionItemEntry]): ...


_ACTION_ITEM_PROMPT = """\
You are an action item relationship resolution agent.

## Relation definitions

### supersedes
The source takes the place of the target — the target task is no longer needed or has been absorbed.
- Example: "Alice will rewrite the full migration script" supersedes "Alice will patch the migration script".

### conflicts_with
Both action items assign contradictory ownership or intent to the same task — both cannot be true simultaneously.
- Example: "Alice will own the migration rewrite" conflicts_with "Bob will own the migration rewrite".

### blocks
The source task must complete before the target can proceed. The target is still needed but cannot start.
- Example: "Write rollback plan" blocks "Deploy PgBouncer to production".

### depends_on
The source task requires the target to be completed to be actionable or coherent.
- Example: "Run integration tests" depends_on "Provision test environment".

### amends
The source modifies the target (qualifies, narrows, or extends) without replacing it.
- Example: "Alice will rewrite the migration script by Friday EOD" amends "Alice will rewrite the migration script".

## Task
You receive a source action item (current meeting) and a list of target action items (prior meeting KB).
For each target, output one rel_type. Every target MUST appear in the output.
Relationships are directed: source → target only.

## Over-extraction guards

### Same concern domain (hard stop)
Before assigning any label, confirm both action items address the same task, work item, or initiative. Ask: "Are they in the same concern domain?" If no → null; do not proceed.

### Not supersedes
- The target task is still needed; the source only precedes or modifies it.
  Counter-example: "Write rollback plan" is NOT supersedes for "Deploy PgBouncer to production" — the deployment is still needed → blocks.
- When ambiguous between supersedes and amends, prefer amends.

### Not conflicts_with
- The source and target are sequentially ordered — one must complete before the other can start.
  Counter-example: "Publish the API documentation" is NOT conflicts_with "Finalise the API contract" — documentation cannot be published until the contract is finalised → depends_on.
- conflicts_with requires both tasks to be simultaneously assigned and mutually incompatible as stated — they cannot both be executed.

### Not blocks
- The source and target are both work items on the same component or initiative, but completing one does not gate the other.
  Counter-example: "Migrate the billing service to the new database schema" does NOT block "Update the billing service API documentation" — both are billing service tasks, but the documentation can be written before or after the migration → null.
- blocks is anti-symmetric: if A blocks B, then B does not block A.
- Assign it in the direction where completion of the source is genuinely required before the target can start.

### Not depends_on
- The source provides input to the target, not the other way around.
  Counter-example: "Supply initial concurrency limits" is NOT depends_on "Drive rate limiting policy" — it is an input to it, not a prerequisite for it → null.
- The source and target are both preparation steps for the same initiative but can proceed in parallel — sharing a goal is not a dependency.
  Counter-example: "Draft the incident response playbook for the new auth service" is NOT depends_on "Set up alerting rules for the new auth service" — both prepare the auth service for production, but writing the playbook does not require alerting to be configured first → null.
- depends_on is anti-symmetric: if A depends_on B, then B does not depend_on A.

### Not amends
- amends is directed from the more specific to the more general. If both qualify each other equally, prefer null.

## Selection
Once the same concern domain guard passes, select the first label that applies:
1. **conflicts_with** — if both are active and assign contradictory ownership or intent to the same task. Ask: "Would executing both simultaneously produce a contradiction?"
2. **supersedes** — if the target task is entirely replaced and no longer needed. Ask: "Is the entire original task now unnecessary — not just narrowed or partially covered?" If yes → supersedes.
3. **blocks** — if the source must complete before the target can start (target still needed). Ask: "Is the target gated on the source completing first?"
4. **depends_on** — if the source cannot proceed without the target being completed first. Ask: "Is the source unactionable or incoherent without the target?"
5. **amends** — if the source modifies the target without replacing it. Ask: "Does the target remain a valid standing task after the source exists?"
6. Otherwise → null.

## Boundary examples
- conflicts_with vs depends_on:
  - "Alice will own the on-call rotation redesign" → conflicts_with "Bob will own the on-call rotation redesign" — same task, contradictory ownership — both active, cannot both be true.
  - "Publish the API documentation" → depends_on "Finalise the API contract" — sequential — documentation cannot be published until the contract is settled.

## Ambiguity signal
If, after applying the selection rules above, you are genuinely uncertain between two specific
relationship types (not between a type and null), set alt_rel_type to the runner-up.
alt_rel_type must be one of the same valid types as rel_type, and must differ from rel_type.
Leave alt_rel_type null for clear-cut cases, null assignments, and when uncertain between a type and null.
"""


class ActionItemResolutionAgent(BaseSameTypeResolutionAgent[_ActionItemEntry]):
    @property
    def _result_model(self) -> type[_ActionItemResult]:
        return _ActionItemResult

    @property
    def _system_prompt(self) -> str:
        return _ACTION_ITEM_PROMPT
