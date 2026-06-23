# Reflective Agents — Design Spec

**Date:** 2026-06-17
**Status:** Implemented

## Overview

This spec covers two related changes to the agent layer:

1. **Rename verification → grounding** — `VerificationAgent` and the surrounding config, exception, and README surface are renamed to reflect what the component actually does: a grounding check (does this quote support this claim?), not a correctness check.
2. **Reflective agents** — an optional agent variant that adds a self-review pass to identification and same-type resolution, catching mistakes the single-pass agent tends to make on borderline cases.

---

## 0. Motivation

### Known limitation

The confidence score and `GroundingAgent` both perform a **grounding check** only: does the supporting quote substantiate the node's claim? Neither checks **type correctness** (is this node actually a Decision, or a preference?) or **structural correctness** (does it satisfy the extraction rules in the identification agent's system prompt?). A node can pass grounding with a high confidence score and still be a misclassification. The same gap exists in same-type resolution: no stage validates that a resolved relationship is correctly typed when two types are genuinely competing.

### Goals

- Make the naming accurate: rename `VerificationAgent` → `GroundingAgent` everywhere.
- Provide an opt-in mechanism for agents to self-review against their own rules, catching errors that grounding cannot.
- Keep the public API identical between shallow and reflective variants so the orchestrator requires no changes.
- Keep the feature strictly opt-in via config flags — no behaviour change unless explicitly enabled.

---

## 1. Rename: Verification → Grounding

Every reference to "verification" that means the quote-grounding check is renamed:

| Old | New |
|---|---|
| `VerificationAgent` | `GroundingAgent` |
| `VerificationRetryExhaustedError` | `GroundingRetryExhaustedError` |
| `VerificationResult` | `GroundingResult` |
| `VerificationLLMConfig` | `GroundingLLMConfig` |
| `ExtractionConfig.verification` | `ExtractionConfig.grounding` |
| `src/seshat/agents/verification.py` | `src/seshat/agents/grounding.py` |

The `GroundingAgent` interface is otherwise unchanged.

---

## 2. Reflective Agents

### 2.1 Concept

A **reflective agent** wraps an existing shallow agent and adds a second LLM pass to catch mistakes the first pass tends to make. Both variants expose the same public method (`identify()` for identification, `resolve()` for resolution), so the orchestrator requires no changes. The second pass is deliberately conservative — it degrades gracefully to shallow behaviour on any failure.

Both are enabled via `ReflectiveLLMConfig` fields on `ExtractionConfig`:

```python
class ReflectiveLLMConfig(BaseConfig):
    enabled: bool = False
    llm: _LLMConfig | None = None  # defaults to the primary LLM when None
```

- `identification_self_review.enabled` — wraps all identification agents in `ReflectiveIdentificationAgent`.
- `resolution_self_review.enabled` — wraps all same-type resolution agents in `ReflectiveResolutionAgent`.

When `enabled` is `False` (default), the registries instantiate shallow agents unchanged.

### 2.2 Class hierarchy

Both reflective agents use the **subclass-and-delegate** (proxy) pattern: they inherit from the same base as the shallow agents, hold an `inner` instance, and delegate all abstract properties to it.

```
_BaseIdentificationAgent[M]
├── DecisionIdentificationAgent        (shallow)
├── RiskIdentificationAgent            (shallow)
├── ActionItemIdentificationAgent      (shallow)
├── OpenQuestionIdentificationAgent    (shallow)
└── ReflectiveIdentificationAgent[M]   (delegates concept_type, output_schema, _system_prompt to inner)

_BaseResolutionAgent[E]
├── BaseSameTypeResolutionAgent[E]
│   ├── DecisionResolutionAgent        (shallow)
│   ├── RiskResolutionAgent            (shallow)
│   ├── ActionItemResolutionAgent      (shallow)
│   └── OpenQuestionResolutionAgent    (shallow)
│   └── ReflectiveResolutionAgent[E]   (wraps same-type agents only — see §2.4)
└── BaseCrossTypeResolutionAgent[E]
    └── ...                            (shallow, no reflective wrapper — see §2.4)
```

### 2.3 `ReflectiveIdentificationAgent` (`identification/reflective.py`)

Wraps any `_BaseIdentificationAgent` in an **extract → validate → filter** pass.

1. **Extract** — delegates to `inner._identify()`, producing `list[AnchoredConcept[M]]`.
2. **Validate** — a single LLM call using `review_llm` that checks each item on two dimensions:
   - *Logical compliance*: does the item satisfy the extraction rules (positive criteria and over-extraction guards)?
   - *Semantic compliance*: does the title/description match what the quote actually contains?
3. **Filter** — items where `passed=False` are discarded. The validator is conservative by design: it rejects only clear rule violations, not borderline quality. When in doubt, it passes.

The validation response model:

```python
class NodeReview(BaseModel):
    passed: bool
    rationale: str | None  # required when passed=False, else None

class SelfReviewResult(BaseModel):
    reviews: list[NodeReview]  # one per extracted item, same order
```

**Graceful degradation:** if the validation call exhausts retries or returns a count mismatch, all extracted nodes are returned as-is (shallow behaviour). If `identify()` returns no nodes, the validation call is skipped entirely.

If `grouped_identification` is enabled for the concept type, grouping runs on the filtered nodes (not on all extracted nodes before filtering).

### 2.4 `ReflectiveResolutionAgent` (`resolution/same_type/reflective.py`)

Wraps any same-type resolution agent with a **competing-hypothesis tiebreaker**.

The key insight from eval analysis was that a blanket validate-and-filter pass over-rejected unambiguous entries, degrading recall without improving precision. The failure mode was concentrated on genuinely borderline cases — pairs where two specific relationship types competed (e.g. `amends` vs `supersedes`). The tiebreaker targets only those cases.

**Mechanism:**

1. The inner agent's entry model gains an optional `alt_rel_type` field. The extractor populates it only when two specific relationship types are genuinely competing for a pair — never for clear-cut cases, never for null assignments, never when uncertain between a type and null.
2. After the inner agent resolves relationships for a source node, entries are split into:
   - **Uncontested** (`alt_rel_type is None`) — returned as-is, no second call.
   - **Contested** (`alt_rel_type is not None`) — sent to a single tiebreaker call.
3. The tiebreaker call receives the contested entries serialised with both candidates, and returns one `chosen` value per entry.
4. The chosen value overwrites `rel_type` on the entry.

The tiebreaker response model:

```python
class TiebreakerEntry(BaseModel):
    chosen: str       # the winning rel_type value
    rationale: str    # one sentence explaining the choice

class TiebreakerResult(BaseModel):
    decisions: list[TiebreakerEntry]  # one per contested entry, same order
```

**Graceful degradation:** on any tiebreaker failure (retries exhausted, count mismatch, invalid `chosen` value), the original `rel_type` from the extractor is kept for the affected entries.

**`alt_rel_type` constraints:** validated by a Pydantic `model_validator` — `alt_rel_type` must differ from `rel_type`. Each concrete entry subclass narrows both `rel_type` and `alt_rel_type` to the same `Literal` of allowed types for that concept, so the extractor cannot signal ambiguity between types that are not both valid for the pair.

**Why same-type only:** eval results showed cross-type F1 = 1.000 under both shallow and reflective modes. There is no quality gap to close for cross-type, so cross-type agents are not wrapped.

### 2.5 Token overhead

| Mode | LLM calls | Approx. input tokens |
|---|---|---|
| Shallow | 96 | ~203k |
| Previous reflective (validate→filter all) | 151 | ~321k |
| Current reflective (tiebreaker on contested only) | 96 | ~209k |

The tiebreaker fires only when the extractor signals ambiguity. In practice it may rarely fire, meaning the overhead is near zero and the benefit comes from the prompt-level signal (`alt_rel_type` field + `## Ambiguity signal` section) forcing more deliberate primary classification.

---

## 3. What this does not change

- The `GroundingAgent` is unchanged in behaviour, interface, and position in the pipeline.
- The orchestrator's grounding/heuristics scoring path is unchanged.
- The `APPROVED / PENDING_REVIEW / REJECTED` status logic is unchanged.
- Resolution post-processing (anti-symmetry, mutual exclusion, superseded-node stripping) runs on the final accepted output.
- Cross-type resolution agents are not wrapped.

---

## 4. File changes

| Action | Path |
|---|---|
| Rename | `src/seshat/agents/verification.py` → `src/seshat/agents/grounding.py` |
| Modify | `src/seshat/config/settings.py` — rename `VerificationLLMConfig` → `GroundingLLMConfig`; add `ReflectiveLLMConfig`, `identification_self_review`, `resolution_self_review` |
| Create | `src/seshat/agents/identification/reflective.py` — `ReflectiveIdentificationAgent`, `SelfReviewResult`, `NodeReview` |
| Create | `src/seshat/agents/resolution/same_type/reflective.py` — `ReflectiveResolutionAgent`, `TiebreakerResult`, `TiebreakerEntry` |
| Modify | `src/seshat/agents/resolution/base.py` — add `alt_rel_type` to `_SameTypeEntry`; change `_run_for_source` return type to `tuple[list[E], dict[str, UUID]]` |
| Modify | `src/seshat/agents/resolution/same_type/{decision,risk,action_item,open_question}.py` — add typed `alt_rel_type` field and `## Ambiguity signal` prompt section |
| Modify | `src/seshat/agents/resolution/cross_type/registry.py` — remove `review_llm` parameter and wrapping logic |
| Modify | `src/seshat/agents/identification/registry.py` — wrap agents when `identification_self_review.enabled` |
| Modify | `src/seshat/agents/resolution/same_type/registry.py` — wrap agents when `resolution_self_review.enabled` |
| Modify | `src/seshat/agents/README.md` — update exception hierarchy, add reflective agents section |

---

## 5. Out of scope

- Per-agent enable/disable flags (the global `enabled` flag per family is sufficient for now).
- Reflective grouping agent — deferred until eval surfaces grouping quality as a problem.
- Reflective cross-type resolution — no quality gap observed in eval.

---

## 6. Design history note

The original design (drafted 2026-06-17) specified a **validate → re-extract loop** for resolution: the validator would check all resolved relationships against the system prompt and trigger a full re-resolution if any failed, cycling up to `max_retries` times before discarding. Eval analysis showed this approach degraded recall without improving precision — every recall loss was on corpus-flagged boundary cases where the validator over-rejected correct relationships. The competing-hypothesis tiebreaker (implemented 2026-06-22) replaced this design: rather than reviewing all output, the mechanism targets only the specific entries the extractor itself flags as ambiguous.
