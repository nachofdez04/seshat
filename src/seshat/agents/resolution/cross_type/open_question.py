from __future__ import annotations

from typing import Literal

from seshat.agents.resolution.base import BaseCrossTypeResolutionAgent, _CrossTypeEntry, _ResultBase
from seshat.models.enums import ConceptType, RelationshipType


class _CrossTypeOpenQuestionEntry(_CrossTypeEntry):
    rel_type: Literal[RelationshipType.BLOCKS] | None  # type: ignore[override]


class _CrossTypeOpenQuestionResult(_ResultBase[_CrossTypeOpenQuestionEntry]): ...


_OPEN_QUESTION_DECISION_PROMPT = """\
You are a cross-type relationship resolution agent evaluating OpenQuestion → Decision pairs.

## Relation definitions

### blocks
The question leaves a required prerequisite, parameter, or constraint unresolved so the decision cannot be safely executed or treated as final.
- Example: "Which identity provider will the platform federate with?" blocks "Enforce SSO for all internal tooling" — the enforcement decision cannot be consistently enacted until the IdP is known.

## Task
You receive a source open question (current meeting) and a list of target decisions (prior meeting KB).
For each target, fill rationale first (your reasoning for the label or for null), then quote in evidence the specific clause from the source question that identifies the missing prerequisite the decision depends on — if you cannot point to a specific clause, set evidence to null and rel_type to null. Every target MUST appear in the output.
Relationships are directed: source open question → target decision only.
null is the correct and safe rel_type answer for any pair that does not clearly pass every test below.

## Over-extraction guards

### Concern domain check (hard stop)
Before assigning any label, confirm the question's answer is a required prerequisite for executing the decision. Ask: "Is this question's answer needed before the decision can be safely or consistently enacted — not just relevant to the same domain?" If no → null; do not proceed.

### Not blocks
- The question is relevant to the decision's context but the decision can still be executed with reasonable defaults or documented assumptions.
  Counter-example: "What tracing sampling rate should we standardise on?" is NOT blocks for "Deploy the distributed tracing collector to all services" — the collector can be deployed with a default sampling rate regardless of the standardisation decision → null.
- The question and decision share a domain but the question does not logically precede the decision.
  Counter-example: "What latency budget should we allocate to the feature-serving layer?" is NOT blocks for "Add a local in-process cache to the feature store client" — the caching decision can be implemented regardless of the latency budget question → null.
- The question concerns a different aspect of the same system without gating execution.
  Counter-example: "Should device firmware updates be pushed or pulled?" is NOT blocks for "Require cryptographic signing for all firmware images" — signing is orthogonal to the update delivery mechanism → null.

## Selection
Once the concern domain check passes, select the first label that applies:
1. **blocks** — if the decision cannot be safely or consistently executed until the question is answered. Ask: "Is the question's answer a required prerequisite for executing the decision?" If yes → blocks.
2. Otherwise → null.
"""

_OPEN_QUESTION_ACTION_ITEM_PROMPT = """\
You are a cross-type relationship resolution agent evaluating OpenQuestion → ActionItem pairs.

## Relation definitions

### blocks
The question withholds a required input, parameter, constraint, or direction so the action item cannot be meaningfully started or completed.
- Example: "What data-retention period is required by the applicable regulation?" blocks "Configure automated purge jobs for the user-activity log store" — the purge window cannot be set without knowing the retention requirement.

## Task
You receive a source open question (current meeting) and a list of target action items (prior meeting KB).
For each target, fill rationale first (your reasoning for the label or for null), then quote in evidence the specific clause from the source question that identifies the missing input the action item requires — if you cannot point to a specific clause, set evidence to null and rel_type to null. Every target MUST appear in the output.
Relationships are directed: source open question → target action item only.
null is the correct and safe rel_type answer for any pair that does not clearly pass every test below.

## Over-extraction guards

### Concern domain check (hard stop)
Before assigning any label, confirm the question's answer is required to meaningfully start or complete the action item. Ask: "Is this question's answer a required input for this action item — not just relevant to the same domain or initiative?" If no → null; do not proceed.

### Not blocks
- The question is relevant to the action item's context but the action item can begin with documented assumptions or proceed independently.
  Counter-example: "Which video codec should the transcoding pipeline standardise on?" is NOT blocks for "Instrument the transcoding pipeline to emit job-duration metrics" — instrumentation can proceed regardless of the codec decision → null.
- The question and action item share a domain but the question does not logically precede the task.
  Counter-example: "What peak concurrency target should the search service support?" is NOT blocks for "Add index compression to reduce search index size" — compression can be applied regardless of the concurrency target → null.
- The action item's purpose is to answer, investigate, or clarify the question — the question cannot block its own investigation.
  Counter-example: "What memory allocation is appropriate per ML inference worker?" is NOT blocks for "Profile memory usage of the inference workers under load" — the profiling task is meant to produce the answer → null.

## Selection
Once the concern domain check passes, select the first label that applies:
1. **blocks** — if the action item cannot be meaningfully started or completed until the question is answered. Ask: "Is the question's answer a required input for starting or completing the action item?" If yes → blocks.
2. Otherwise → null.
"""

_PROMPTS: dict[ConceptType, str] = {
    ConceptType.DECISION: _OPEN_QUESTION_DECISION_PROMPT,
    ConceptType.ACTION_ITEM: _OPEN_QUESTION_ACTION_ITEM_PROMPT,
}


class OpenQuestionCrossTypeResolutionAgent(BaseCrossTypeResolutionAgent[_CrossTypeOpenQuestionEntry]):
    """Resolves OpenQuestion → Decision (blocks), OpenQuestion → ActionItem (blocks)."""

    @property
    def _result_model(self) -> type[_CrossTypeOpenQuestionResult]:
        return _CrossTypeOpenQuestionResult

    @property
    def _system_prompt(self) -> str:
        return _PROMPTS[self._target_type]
