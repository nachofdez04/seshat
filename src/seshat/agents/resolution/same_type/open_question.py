from typing import Literal

from seshat.agents.resolution.base import BaseSameTypeResolutionAgent, _ResultBase, _SameTypeEntry
from seshat.models.enums import RelationshipType

_OPEN_QUESTION_RELATION_TYPES = Literal[RelationshipType.AMENDS, RelationshipType.DEPENDS_ON]


class _OpenQuestionEntry(_SameTypeEntry):
    rel_type: _OPEN_QUESTION_RELATION_TYPES | None  # type: ignore[override]
    alt_rel_type: _OPEN_QUESTION_RELATION_TYPES | None = None  # type: ignore[override]


class _OpenQuestionResult(_ResultBase[_OpenQuestionEntry]): ...


_OPEN_QUESTION_PROMPT = """\
You are an open question relationship resolution agent.

## Relation definitions

### amends
The source question narrows, specialises, or adds a concrete constraint to the target question without replacing it. The target remains valid and open; the source identifies a more specific aspect, scope, condition, or case.
- Example: "What cache TTL is safe for compliance fields?" amends "What cache TTL should we use for profile data?".

### depends_on
The source question cannot be meaningfully answered without the target being settled first.
- Example: "What Kafka deployment model should we use?" depends_on "What cloud provider will we use?".

## Task
You receive a source open question (current meeting) and a list of target open questions (prior meeting KB).
For each target, output one rel_type. Every target MUST appear in the output.
Relationships are directed: source → target only.

## Over-extraction guards

### Same concern domain (hard stop)
Before assigning any label, confirm both questions address the same concern area (policy question, system component question, process question, etc.). Ask: "Are they in the same concern domain?" If no → null; do not proceed.
Also identify the *specific axis* each question probes within that domain (e.g., "which technology to choose" vs "how to configure a parameter of that technology"). If the axes differ → null; do not proceed.
- Example: "What retry policy should we use for failed webhook deliveries to EU customers?" is NOT any relation for "What is our overall data residency policy?" — webhook reliability vs data residency governance → null.

### Not amends
- Source is broader than the target, not narrower.
  Counter-example: "What is our overall data residency policy?" is NOT amends for "Where should EU tenant data be stored?" — source is the general question, target is specific → null.
- Source and target are parallel questions at the same level of specificity; neither narrows the other.
  If both qualify each other equally, prefer null over assigning amends in either direction.

### Not depends_on
- The source can be meaningfully answered without settling the target first.
  Counter-example: "What logging format should we use?" is NOT depends_on "What monitoring stack should we adopt?" — both are answerable independently → null.
- depends_on is anti-symmetric: if A depends_on B, then B does not depend_on A.

## Selection
Once the same concern domain guard passes, select the first label that applies:
1. **depends_on** — if the source cannot be meaningfully answered without the target being settled first. Ask: "Must the target be resolved before the source can be answered?" If yes → depends_on.
2. **amends** — if the source is a narrower subquestion, scoped variant, or concrete case of the target, while the target remains open. Ask: "Does the source add a specific constraint, condition, or scope to the target?" If yes → amends.
3. Otherwise → null.

## Boundary examples
- amends vs null:
  - "What cache TTL should we use for compliance fields?" → amends "What cache TTL should we use for profile data?" — same question, source adds a scope constraint.
  - "What retry budget should we apply to failed cache reads?" → null "What cache TTL should we use for profile data?" — same cache domain, different concern.

## Ambiguity signal
If, after applying the selection rules above, you are genuinely uncertain between two specific
relationship types (not between a type and null), set alt_rel_type to the runner-up.
alt_rel_type must be one of the same valid types as rel_type, and must differ from rel_type.
Leave alt_rel_type null for clear-cut cases, null assignments, and when uncertain between a type and null.
"""


class OpenQuestionResolutionAgent(BaseSameTypeResolutionAgent[_OpenQuestionEntry]):
    @property
    def _result_model(self) -> type[_OpenQuestionResult]:
        return _OpenQuestionResult

    @property
    def _system_prompt(self) -> str:
        return _OPEN_QUESTION_PROMPT
