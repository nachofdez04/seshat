from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from seshat.agents.resolution.base import BaseCrossTypeResolutionAgent, _CrossTypeEntry, _ResultBase
from seshat.models.enums import ConceptType, RelationshipType

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from seshat.config.settings import ResolutionLLMConfig


class _CrossTypeRiskEntry(_CrossTypeEntry):
    rel_type: Literal[RelationshipType.BLOCKS] | None  # type: ignore[override]


class _CrossTypeRiskResult(_ResultBase[_CrossTypeRiskEntry]): ...


_RISK_DECISION_PROMPT = """\
You are a cross-type relationship resolution agent evaluating Risk → Decision pairs.

## Relation definitions

### blocks
The risk, if unresolved, makes it impossible or definitionally wrong to act on the decision as stated.
- Example: "Export classification ruling for this component is still pending with the trade-compliance office" blocks "Ship the updated hardware module to the overseas manufacturing partner" — the decision cannot be legally enacted until the ruling is known.

## Task
You receive a source risk (current meeting) and a list of target decisions (prior meeting KB).
For each target, fill rationale first (your reasoning for the label or for null), then quote in evidence the specific clause from the source risk that states the unresolved condition that makes enacting the decision impossible — if you cannot point to a specific clause, set evidence to null and rel_type to null. Every target MUST appear in the output.
Relationships are directed: source risk → target decision only.
null is the correct and safe rel_type answer for any pair that does not clearly pass every test below.

## Over-extraction guards

### Concern domain check (hard stop)
Before assigning any label, confirm the risk directly threatens the feasibility of the decision. Ask: "Does this unresolved risk make it impossible or definitionally wrong to enact this decision — not just risky or suboptimal?" If no → null; do not proceed.

### Not blocks
- The risk describes a concern, consequence, or tradeoff related to the decision without making it impossible or definitionally wrong to act on.
  Counter-example: "The chosen open-source library has no commercial support tier" is NOT blocks for "Standardise on that library for all internal data-transformation jobs" — the support gap is a tradeoff to manage, not a concrete obstacle to adopting it → null.
- The risk motivates, informs, or is addressed by the decision without preventing execution.
  Counter-example: "Firmware rollback capability not yet validated on the new hardware revision" is NOT blocks for "Defer the firmware upgrade until rollback has been validated" — the risk motivates the deferral but does not prevent executing it → null.
- The risk is a symptom of the current situation, not a prerequisite the decision depends on.
  Counter-example: "Teams may misapply the new expense-approval policy during the transition period" is NOT blocks for "Adopt the revised expense-approval policy company-wide" — the risk describes a rollout hazard; the decision can be adopted regardless → null.

## Selection
Once the concern domain check passes, select the first label that applies:
1. **blocks** — the risk makes it impossible or definitionally wrong to act on the decision as stated. Ask: "Does the unresolved risk make it impossible or definitionally wrong to act on this decision?" If yes → blocks.
2. Otherwise → null.

## Boundary examples
- blocks vs null:
  - "Load test shows the new message broker cannot sustain the required throughput at peak" → blocks "Migrate all event publishing to the new message broker" — executing the decision would concretely fail.
  - "The new message broker may experience elevated latency under sustained load" → null "Standardise on the new message broker for non-critical notification events" — risk is a concern; the decision can still be adopted.

## Positive examples
- "Regulatory sandbox approval for the new product feature is still outstanding" blocks "Launch the new product feature to all customers" — the decision cannot be executed without the approval.
"""

_RISK_OPEN_QUESTION_PROMPT = """\
You are a cross-type relationship resolution agent evaluating Risk → OpenQuestion pairs.

## Relation definitions

### blocks
The risk makes every possible answer to the question premature or definitionally invalid until it is resolved. The risk withholds a constraint, fact, or regulatory outcome that determines what answers are even permissible — not just one factor to weigh.
- Example: "Compliance audit may prohibit storing session tokens longer than 24 h" blocks "What session token TTL should we use?" — every TTL answer is invalid until the audit outcome is known.

## Task
You receive a source risk (current meeting) and a list of target open questions (prior meeting KB).
For each target, fill rationale first (your reasoning for the label or for null), then quote in evidence the specific clause from the source risk that states the unresolved condition that makes every possible answer to the question premature — if you cannot point to a specific clause, set evidence to null and rel_type to null. Every target MUST appear in the output.
Relationships are directed: source risk → target open question only.
null is the correct and safe rel_type answer for any pair that does not clearly pass every test below.

## Over-extraction guards

### Concern domain check (hard stop)
Before assigning any label, confirm the risk withholds information that determines what answers to the question are even permissible. Ask: "Does this risk make every possible answer to the question premature or definitionally invalid — not just harder to evaluate, less certain, or ruling out one candidate while leaving defensible alternatives?" If no → null; do not proceed.

### Not blocks
- The risk informs or motivates the question but a defensible answer can still be given despite the risk.
  Counter-example: "A managed search vendor may not meet latency targets" is NOT blocks for "Should we use hosted, self-managed, or database-native search?" — the risk informs one option but the question can still be analysed and answered → null.
- The risk is a symptom of the situation — the question can be answered regardless of whether the risk is resolved.
  Counter-example: "Engineers unfamiliar with the new auth service may escalate incidents incorrectly" is NOT blocks for "Should we consolidate on-call ownership across all services?" — the risk follows from expansion decisions; it doesn't gate the ownership question → null.
- The risk rules out one or more candidate answers but a defensible answer can still be given from the remaining options.
  Counter-example: "Refresh tokens stored in browser localStorage are vulnerable to XSS exfiltration" is NOT blocks for "What is the recommended client-side token storage pattern?" — the risk rules out localStorage, but a defensible answer (e.g., httpOnly cookies) can still be given without resolving the risk → null.

## Selection
Once the concern domain check passes, select the first label that applies:
1. **blocks** — only if every possible answer to the question would be premature or definitionally invalid until the risk is resolved. Ask: "Can a defensible answer be given to this question despite the risk?" If yes → null. If no → blocks.
2. Otherwise → null.

## Boundary examples
- blocks vs null:
  - "Load test shows payment gateway cannot sustain peak checkout traffic" → blocks "Should we route all checkouts through the new gateway this quarter?" — the performance data makes every answer to the question invalid until resolved.
  - "Third-party vendor may not meet SLA guarantees" → null "Which vendor should we use for log aggregation?" — risk informs the evaluation; a defensible answer can still be given.

## Positive examples
- "Legal review still pending on data residency" blocks "Where should EU tenant data be hosted?" — no valid answer exists until the legal constraint is known.
"""

_RISK_ACTION_ITEM_PROMPT = """\
You are a cross-type relationship resolution agent evaluating Risk → ActionItem pairs.

## Relation definitions

### blocks
The risk makes the action item impossible to complete correctly as stated — proceeding would produce incorrect or harmful results, or the task literally cannot be executed until the risk is resolved.
- Example: "The new labelling schema contains unresolved conflicts between two taxonomy working groups" blocks "Re-label the entire training corpus using the new labelling schema" — executing would embed the conflicting labels into the corpus.

## Task
You receive a source risk (current meeting) and a list of target action items (prior meeting KB).
For each target, fill rationale first (your reasoning for the label or for null), then quote in evidence the specific clause from the source risk that states the unresolved condition that makes completing the action item impossible — if you cannot point to a specific clause, set evidence to null and rel_type to null. Every target MUST appear in the output.
Relationships are directed: source risk → target action item only.
null is the correct and safe rel_type answer for any pair that does not clearly pass every test below.

## Over-extraction guards

### Concern domain check (hard stop)
Before assigning any label, confirm the risk directly threatens the correctness or executability of the action item. Ask: "Would proceeding with this action item while this risk is unresolved produce incorrect or harmful results, or make it impossible to complete?" If no → null; do not proceed.

### Not blocks
- The risk raises concerns relevant to the action item's domain but the task can still be executed or started.
  Counter-example: "The new supplier's lead times may increase during peak season" is NOT blocks for "Audit current inventory levels and flag items below safety stock" — same procurement domain, but the auditing task is not obstructed → null.
- The action item's purpose is to investigate, validate, or mitigate the risk — the risk cannot block its own resolution work.
  Counter-example: "Batch job scheduling conflicts may cause data gaps in the nightly report" is NOT blocks for "Map all batch job dependencies and identify scheduling conflicts" — the action item is meant to investigate the risk, not be blocked by it → null.
- The risk makes the action item's outcome less certain or more complex but the task can still proceed.
  Counter-example: "Legacy ERP integration may produce duplicate records during the cutover window" is NOT blocks for "Update the field-mapping configuration for the new ERP connector" — the duplication risk is a concern during cutover; the configuration update can proceed independently → null.

## Selection
Once the concern domain check passes, select the first label that applies:
1. **blocks** — the action item cannot be completed correctly until the risk is resolved. Ask: "Would proceeding with this action item while the risk is unresolved produce incorrect or harmful results, or make the task impossible to execute?" If yes → blocks.
2. Otherwise → null.

## Positive examples
- "The reference dataset used for model validation contains a known data-quality error that skews precision metrics" blocks "Publish the model validation report using the reference dataset" — the report would be built on incorrect data.
- "Stress test shows the candidate infrastructure cannot handle the required concurrent user load" blocks "Set the platform's advertised concurrent user capacity to the target figure" — the action would publish a figure that contradicts measured reality.
"""

_PROMPTS: dict[ConceptType, str] = {
    ConceptType.DECISION: _RISK_DECISION_PROMPT,
    ConceptType.OPEN_QUESTION: _RISK_OPEN_QUESTION_PROMPT,
    ConceptType.ACTION_ITEM: _RISK_ACTION_ITEM_PROMPT,
}


class RiskCrossTypeResolutionAgent(BaseCrossTypeResolutionAgent[_CrossTypeRiskEntry]):
    """Resolves Risk → Decision (blocks), Risk → OpenQuestion (blocks), Risk → ActionItem (blocks)."""

    def __init__(self, llm: BaseChatModel, config: ResolutionLLMConfig, target_type: ConceptType) -> None:
        super().__init__(llm=llm, config=config)
        self._target_type = target_type

    @property
    def _result_model(self) -> type[_CrossTypeRiskResult]:
        return _CrossTypeRiskResult

    @property
    def _system_prompt(self) -> str:
        return _PROMPTS[self._target_type]
