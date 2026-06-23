from __future__ import annotations

from typing import Literal

from seshat.agents.resolution.base import BaseCrossTypeResolutionAgent, _CrossTypeEntry, _ResultBase
from seshat.models.enums import ConceptType, RelationshipType


class _CrossTypeActionItemEntry(_CrossTypeEntry):
    rel_type: Literal[RelationshipType.MITIGATES] | None  # type: ignore[override]


class _CrossTypeActionItemResult(_ResultBase[_CrossTypeActionItemEntry]): ...


_ACTION_ITEM_RISK_PROMPT = """\
You are a cross-type relationship resolution agent evaluating ActionItem → Risk pairs.

## Relation definitions

### mitigates
Completing the action item directly implements or operationalises a concrete control that reduces the risk's likelihood, severity, exposure, or blast radius. The mitigation must be an expected effect of completing the task itself.
- Example: "Configure automated model-drift alerts with a rollback trigger" mitigates "Silent model drift may degrade prediction quality before anyone notices".

## Task
You receive a source action item (current meeting) and a list of target risks (prior meeting KB).
For each target, fill rationale first (your reasoning for the label or for null), then quote in evidence the specific clause from the source action item that directly deploys or enforces the control — if you cannot point to a specific clause, set evidence to null and rel_type to null. Every target MUST appear in the output.
Relationships are directed: source action item → target risk only.
null is the correct and safe rel_type answer for any pair that does not clearly pass every test below.

## Over-extraction guards

### Concern domain check (hard stop)
Before assigning any label, confirm the action item directly addresses the risk's specific failure mode. Ask: "Does completing this task itself deploy, enable, or enforce a control that reduces this risk's failure mode — not just relate to the same system?" If no → null; do not proceed.

### Not mitigates
- The action item investigates, discusses, plans, proposes, documents, monitors, or alerts about the risk without deploying, enabling, or enforcing a concrete control.
  Counter-example: "Assess feasibility of adding input-validation hooks to the ingestion pipeline" is NOT mitigates for "Malformed records in the ingestion pipeline may corrupt the downstream feature store" — feasibility assessment informs the risk but does not reduce it → null.
- The action item creates a plan, proposal, design, or ticket breakdown — it does not mitigate until the controls described are actually deployed or enforced.
  Counter-example: "Draft an architecture proposal for isolating the inference serving layer" is NOT mitigates for "A noisy-neighbour workload on the inference host may cause latency spikes for all tenants" — the proposal describes future controls; the isolation is not yet in place → null.
- The action item is related to the same system or project but does not address the risk's specific failure mode.
  Counter-example: "Update the SSO integration test suite for the new IdP" is NOT mitigates for "Device firmware signing key may be exposed if the CI secrets store is compromised" — same engineering area, but the test update does not reduce secret-exposure risk → null.
- The action item is a step in a broader initiative and completing it advances that initiative, but the step itself does not directly deploy or enforce a control that addresses the risk's failure mode.
  Counter-example: "Onboard the payments service to the new platform" is NOT mitigates for "Mixed migration state may cause inconsistent behaviour across services during the transition" — onboarding one service advances the migration but does not reduce the inconsistency risk; the risk persists until the migration is complete → null.

## Selection
Once the concern domain check passes, select the first label that applies:
1. **mitigates** — completing the action item directly implements a control that reduces the risk's failure mode. Ask: "Does completing this task itself deploy, enable, or enforce a concrete control that reduces the risk?" If yes → mitigates.
2. Otherwise → null.

## Positive examples
- "Enable write-ahead log archiving for the replica promotion workflow" mitigates "Replica promotion without WAL archiving may leave the standby in an inconsistent state" — enabling WAL archiving directly deploys the missing control whose absence is the failure mode.
- "Add cold-storage retrieval path for archived documents" mitigates "Short document lifecycle policies may break archived document retrieval workflows" — the retrieval path directly addresses the workflow failure.
"""


_PROMPTS: dict[ConceptType, str] = {
    ConceptType.RISK: _ACTION_ITEM_RISK_PROMPT,
}


class ActionItemCrossTypeResolutionAgent(BaseCrossTypeResolutionAgent[_CrossTypeActionItemEntry]):
    """Resolves ActionItem → Risk (mitigates)."""

    @property
    def _result_model(self) -> type[_CrossTypeActionItemResult]:
        return _CrossTypeActionItemResult

    @property
    def _system_prompt(self) -> str:
        return _PROMPTS[self._target_type]
