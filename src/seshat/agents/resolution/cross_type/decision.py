from __future__ import annotations

from typing import Literal

from seshat.agents.resolution.base import BaseCrossTypeResolutionAgent, _CrossTypeEntry, _ResultBase
from seshat.models.enums import ConceptType, RelationshipType


class _DecisionToRiskEntry(_CrossTypeEntry):
    rel_type: Literal[RelationshipType.MITIGATES] | None  # type: ignore[override]


class _DecisionToOpenQuestionEntry(_CrossTypeEntry):
    rel_type: Literal[RelationshipType.RESOLVES] | None  # type: ignore[override]


class _DecisionToActionItemEntry(_CrossTypeEntry):
    rel_type: Literal[RelationshipType.BLOCKS] | None  # type: ignore[override]


class _DecisionToRiskResult(_ResultBase[_DecisionToRiskEntry]): ...


class _DecisionToOpenQuestionResult(_ResultBase[_DecisionToOpenQuestionEntry]): ...


class _DecisionToActionItemResult(_ResultBase[_DecisionToActionItemEntry]): ...


_DecisionResult = _DecisionToRiskResult | _DecisionToOpenQuestionResult | _DecisionToActionItemResult


_DECISION_RISK_PROMPT = """\
You are a cross-type relationship resolution agent evaluating Decision → Risk pairs.

## Relation definitions

### mitigates
The decision establishes a policy, architecture, constraint, or control that directly reduces the risk's likelihood, severity, exposure, or blast radius. The mechanism must directly address the risk's specific failure mode.
- Example: "Enforce a minimum two-source approval policy for all critical component orders" mitigates "Single-source procurement dependency may halt production if the primary supplier fails to deliver".

## Task
You receive a source decision (current meeting) and a list of target risks (prior meeting KB).
For each target, fill rationale first (your reasoning for the label or for null), then quote in evidence the specific clause from the source decision that establishes the control or policy that reduces the risk's failure mode — if you cannot point to a specific clause, set evidence to null and rel_type to null. Every target MUST appear in the output.
Relationships are directed: source decision → target risk only.
null is the correct and safe rel_type answer for any pair that does not clearly pass every test below.

## Over-extraction guards

### Concern domain check (hard stop)
Before assigning any label, confirm the decision's policy or control directly addresses the risk's specific failure mode while both are simultaneously active. Ask: "Does this decision's mechanism directly reduce this risk's failure mode — not just share the same system or domain, and not merely make the risk obsolete by retiring its underlying mechanism?" If no → null; do not proceed.

### Not mitigates
- The decision and risk share a domain but the decision addresses a different failure mode.
  Counter-example: "Require code-owners approval on all pull requests to the model training module" is NOT mitigates for "Feature engineering pipeline may silently discard rows with missing sensor readings" — the approval gate targets code-change quality, not the data-loss failure mode → null.
- The decision only detects, monitors, or alerts on the risk without reducing the failure mode itself.
  Counter-example: "Publish a weekly supplier spend report for the procurement team" is NOT mitigates for "Single-source procurement for a critical component may leave production exposed if the supplier fails" — the report increases visibility but does not reduce the single-source dependency → null.
- The decision defers, acknowledges, or postpones work without introducing a concrete control.
  Counter-example: "Postpone the background-check tooling decision until the HR platform review is complete" is NOT mitigates for "Inconsistent background-check processes may create compliance liability" — deferral does not introduce a consistent process → null.
- The decision eliminates or replaces the mechanism that gives rise to the risk, making the risk obsolete as a side effect rather than directly addressing its failure mode.
  Counter-example: "Replace session-token authentication with OAuth 2.0 across all services" is NOT mitigates for "Long-lived session tokens increase the exposure window if intercepted" — the decision retires session tokens entirely, making the risk moot, but does not directly reduce the token-lifetime failure mode while both are simultaneously active → null.

## Selection
Once the concern domain check passes, select the first label that applies:
1. **mitigates** — the decision directly removes, reduces, contains, or controls the specific failure mode that motivated the risk. Ask: "Does the decision mechanistically address this risk's failure mode?" If yes → mitigates.
2. Otherwise → null.

## Positive examples
- "Require two independent reviewers before merging any change to the model scoring pipeline" mitigates "A single reviewer may approve a defect that silently degrades prediction quality" — the review gate directly addresses the single-reviewer failure mode.
- "Set hard memory limits per image-processing worker" mitigates "Image-processing workers may exhaust available memory on oversized uploads" — the limit directly contains the failure mode by capping resource consumption.
"""

_DECISION_OPEN_QUESTION_PROMPT = """\
You are a cross-type relationship resolution agent evaluating Decision → OpenQuestion pairs.

## Relation definitions

### resolves
The decision gives a direct, complete, and final-enough answer to the open question so it should no longer be tracked as open.
- Example: "Use contractor performance scorecards reviewed quarterly" resolves "How should we evaluate contractor performance?".

## Task
You receive a source decision (current meeting) and a list of target open questions (prior meeting KB).
For each target, fill rationale first (your reasoning for the label or for null), then quote in evidence the specific clause from the source decision that directly answers the question — if you cannot point to a specific clause, set evidence to null and rel_type to null. Every target MUST appear in the output.
Relationships are directed: source decision → target open question only.
null is the correct and safe rel_type answer for any pair that does not clearly pass every test below.

## Over-extraction guards

### Concern domain check (hard stop)
Before assigning any label, confirm the decision directly addresses what the question is asking. Ask: "Does this decision's answer directly settle what this question asks — not just relate to the same system or project?" If no → null; do not proceed.

### Not resolves
- The decision answers only a subcase or instance of the question, or narrows it without fully settling it.
  Counter-example: "Adopt Kubernetes for the batch inference workloads" is NOT resolves for "What container orchestration approach should we use across the whole platform?" — the decision covers one workload class, not the platform-wide question → null.
- The decision is phased, provisional, or temporary and the question is broader in scope.
  Counter-example: "Pilot the new onboarding workflow with the Berlin office only" is NOT resolves for "Should we standardise the onboarding workflow across all offices?" — the pilot does not settle the company-wide question → null.
- The decision assumes or presupposes an answer to the question rather than providing one.
  Counter-example: "Configure the anomaly detector to use the gradient-boosting model" is NOT resolves for "Which ML framework should the anomaly detection team standardise on?" — the decision presupposes a model choice but does not answer the framework question → null.

## Selection
Once the concern domain check passes, select the first label that applies:
1. **resolves** — if the decision provides a direct, complete answer that closes the open question. Ask: "Does this decision fully settle the question so it no longer needs to be tracked as open?" If yes → resolves.
2. Otherwise → null.

## Positive examples
- "Source all raw materials exclusively through pre-approved vendors on the central procurement list" resolves "Which vendors are authorised for raw-material procurement?" — the decision gives a complete, actionable answer that closes the question.
- "Use the internal feature-flag service for all A/B experiments going forward" resolves "What tooling should teams use to run A/B experiments?" — the decision settles the tooling choice so the question no longer needs to be tracked as open.
"""

_DECISION_ACTION_ITEM_PROMPT = """\
You are a cross-type relationship resolution agent evaluating Decision → ActionItem pairs.

## Relation definitions

### blocks
The decision imposes a restriction, freeze, prohibition, or incompatible constraint that prevents the action item from proceeding as stated. If the action item can proceed by incorporating the decision as a new constraint or updating its deliverable, use null.
- Example: "Suspend all third-party data-sharing agreements pending the privacy audit" blocks "Send aggregated usage data to the analytics partner".

## Task
You receive a source decision (current meeting) and a list of target action items (prior meeting KB).
For each target, fill rationale first (your reasoning for the label or for null), then quote in evidence the specific clause from the source decision that imposes the prohibition or freeze that makes execution impossible — if you cannot point to a specific clause, set evidence to null and rel_type to null. Every target MUST appear in the output.
Relationships are directed: source decision → target action item only.
null is the correct and safe rel_type answer for any pair that does not clearly pass every test below.

## Over-extraction guards

### Concern domain check (hard stop)
Before assigning any label, confirm the decision directly governs what the action item does. Ask: "Does this decision impose a specific prohibition, freeze, or incompatible constraint that makes executing this action item impossible as written — not just pointless, superseded, or redundant because the target system is being retired or replaced, or because the decision has already made the choice the action was meant to produce?" If no → null; do not proceed.

### Not blocks
- The decision changes the context or parameters of the action item but the action item can still proceed by incorporating that change.
  Counter-example: "Increase the minimum re-order quantity for all components to 500 units" is NOT blocks for "Review Q3 component inventory and flag shortfalls" — the decision changes the threshold but doesn't prevent the review → null.
- The decision makes the action item redundant or obsolete by replacing the system or tool the action item was targeting — "no longer needed" is not the same as "prevented from executing."
  Counter-example: "Retire the legacy HR portal and migrate to the new platform" is NOT blocks for "Add the new-hire checklist to the legacy HR portal" — the task is no longer useful, but there is no constraint preventing it from being executed as stated → null.
- The decision shares the same domain or system but does not directly prevent execution.
  Counter-example: "Require all firmware releases to pass hardware-in-the-loop testing" is NOT blocks for "Update the firmware build toolchain to the latest compiler version" — same firmware domain, but the testing requirement does not prevent the toolchain update → null.
- The decision makes the choice that the action item was supposed to produce, rendering the action item obsolete — but obsolescence is not a prohibition.
  Counter-example: "Adopt PostgreSQL as the standard database for all new services" is NOT blocks for "Evaluate PostgreSQL, MySQL, and CockroachDB and recommend a database" — the decision has already made the choice the action item was meant to produce, but nothing prevents the evaluation from proceeding → null.

## Selection
Once the concern domain check passes, select the first label that applies:
1. Ask: could someone execute this action item right now if they chose to? If yes but it would be pointless or wasteful, use null — blocks requires that execution is physically, legally, or logically impossible, not merely unnecessary.
2. **blocks** — if the decision imposes a restriction or freeze that makes execution impossible as stated. Ask: "Does this decision impose a restriction, freeze, or prohibition that prevents the action item from proceeding as stated?" If yes → blocks.
3. Otherwise → null.

## Positive examples
- "Halt all external vendor onboarding until the procurement audit is complete" blocks "Onboard the new translation services vendor" — the halt is an explicit prohibition that makes onboarding impossible to proceed as stated.
- "Require board sign-off before committing to any multi-year infrastructure contract" blocks "Sign the three-year data-centre colocation agreement" — the sign-off requirement is an incompatible constraint that prevents execution without prior approval.
"""

_PROMPTS: dict[ConceptType, str] = {
    ConceptType.RISK: _DECISION_RISK_PROMPT,
    ConceptType.OPEN_QUESTION: _DECISION_OPEN_QUESTION_PROMPT,
    ConceptType.ACTION_ITEM: _DECISION_ACTION_ITEM_PROMPT,
}


_RESULT_MODELS: dict[ConceptType, type[_DecisionResult]] = {
    ConceptType.RISK: _DecisionToRiskResult,
    ConceptType.OPEN_QUESTION: _DecisionToOpenQuestionResult,
    ConceptType.ACTION_ITEM: _DecisionToActionItemResult,
}


class DecisionCrossTypeResolutionAgent(BaseCrossTypeResolutionAgent):
    """Resolves Decision → Risk (mitigates), Decision → OpenQuestion (resolves), Decision → ActionItem (blocks)."""

    @property
    def _result_model(self) -> type[_DecisionResult]:
        return _RESULT_MODELS[self._target_type]

    @property
    def _system_prompt(self) -> str:
        return _PROMPTS[self._target_type]
