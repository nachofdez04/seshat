from typing import Literal

from seshat.agents.resolution.base import BaseSameTypeResolutionAgent, _ResultBase, _SameTypeEntry
from seshat.models.enums import RelationshipType


class _RiskEntry(_SameTypeEntry):
    rel_type: Literal[RelationshipType.AMENDS] | None  # type: ignore[override]


class _RiskResult(_ResultBase[_RiskEntry]): ...


_RISK_PROMPT = """\
You are a risk relationship resolution agent.

## Relation definitions

### amends
The source refines the target risk without replacing it as a tracked concern.
- Example: "Pipeline may fail for messages above 512 KB" amends "Pipeline may fail for large messages".

## Task
You receive a source risk (current meeting) and a list of target risks (prior meeting KB).
For each target, output one rel_type. Every target MUST appear in the output.
Relationships are directed: source → target only.

## Over-extraction guards

### Same concern domain (hard stop)
Before assigning any label, confirm both risks address the same concern area (failure mode, system component, or risk category). Ask: "Are they in the same concern domain?" If no → null; do not proceed.
- Example: "Connection pool exhausted under high read load" is NOT any relation for "Write-ahead log fills during bulk import" — same database, different mechanisms → null.

### Not amends
- Source and target share the same component or domain but describe different failure modes.
  Counter-example: "Connection pool exhausted under high read load" is NOT amends for "Write-ahead log fills during bulk import" — same database, different mechanisms → null.
- Source is a parallel concern at the same level of precision, not a refinement of the target.
  Counter-example: "Cache stampede on cold start may spike database load" is NOT amends for "Connection pool exhausted under peak traffic" — related domain, independent failure modes → null.
- Both describe the same concern at the same specificity level; neither refines the other.
  If it is unclear which is more specific, prefer null over assigning amends in either direction.

## Selection
Once the same concern domain guard passes, select the first label that applies:
1. **amends** — when source and target describe the same failure mode or concern, and the source refines, corrects, or adds precision to the target. The source may:
   - Narrow the trigger condition: "above 512 KB" amends "for large messages"
   - Identify a more specific failure scenario: "during peak traffic" amends "under load"
   - Correct or update the framing while preserving the same concern: "token exhaustion is most likely at peak traffic, not at cutover" amends "token exhaustion may occur at cutover"
   - Quantify or add a concrete condition to an abstract statement
   Ask: "Does the source add precision to the same underlying failure mode?" If yes → amends.
2. Otherwise → null.

## Boundary examples
- amends vs null:
  - "Pipeline may fail for messages above 512 KB" → amends "Pipeline may fail for large messages" — same failure mode, source adds a threshold.
  - "Pipeline may fail for messages above 512 KB" → null "Pipeline may exhaust memory on very large file uploads" — different failure modes: message routing vs memory.
"""


class RiskResolutionAgent(BaseSameTypeResolutionAgent[_RiskEntry]):
    @property
    def _result_model(self) -> type[_RiskResult]:
        return _RiskResult

    @property
    def _system_prompt(self) -> str:
        return _RISK_PROMPT
