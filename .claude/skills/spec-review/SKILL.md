---
name: spec-review
description: Use when asked to review, critique, or improve a spec, design document, or RFC — surfaces missing rationale, ambiguities, edge cases, contradictions, and what the spec omits entirely
---

# Spec Review

## Overview

A spec review has two jobs: find problems in what's written, and name what's missing entirely. Output severity-first so the author knows where to focus.

## Review Checklist

### 1. Context & Scope
- Is the problem **clearly stated** (not just described)?
- Are **goals and non-goals** defined?
- Is there a **success criterion** — how will anyone know this shipped correctly?

### 2. Rationale & Trade-offs
- Is there a **"why this approach"** section or equivalent?
- Are **alternatives** named and rejected with reasons?
- Are **assumptions** made explicit?

### 3. What's Missing Entirely
Explicitly ask: what should be here that isn't?
- **Acceptance criteria** per feature/finding
- **Failure modes** and recovery strategy
- **Data consistency model** (especially for distributed writes)
- **Performance / cost bounds** (e.g., "max depth 3 — but why?")
- **Validation plan** — how will findings be tested?
- **Reproducibility** — can developers reproduce failures locally, or only in CI/prod?
- **Latency / throughput constraints** — does the design have performance non-goals?

### 4. Edge Cases & Failure Handling
- Does the spec cover failure paths, not just the happy path?
- What happens on **partial failure** (crash mid-operation)?
- What happens when **inputs are absent, empty, or malformed**?

### 5. Contradictions, Inconsistencies & Duplication
- Does the spec say one thing in one place and another elsewhere?
- Does it duplicate information, creating a consistency risk?
- Does it defer decisions it claims to have made?

### 6. Ambiguities & Vague Language
- Are terms defined, or left to the implementor?
- Are thresholds, weights, or heuristics **pinned** or deferred?
- Are function names used that are never specified?

### 7. Anti-patterns
- **Code instead of spec**: pseudocode that depends on undefined functions looks complete but isn't
- **No rationale**: "what" without "why" means reviewers can't evaluate soundness
- **Overengineering upfront**: added complexity without measured cost/benefit
- **Happy-path only**: designs that only work when nothing goes wrong

### 8. Open Questions & Risks
- What **unknowns** remain? Are they acknowledged?
- What are the **top 2–3 risks** if this ships as-is?

## Output Format

Structure output in this order — most critical findings first:

```
## Critical Issues
[Contradictions, missing acceptance criteria, consistency risks — things that block implementation]

## High-Priority Issues
[Underspecified decisions, missing edge cases, unjustified complexity]

## Medium-Priority Issues
[Ambiguities, vague language, missing context]

## Strengths
[What the spec does well — required, not optional]

## Clarifying Questions
[Concrete questions the author must answer before implementation]

## Proposed Simplifications
[Optional: ways to reduce scope or complexity without losing value]
```

## Common Mistakes to Avoid

| Mistake | Reality |
|---------|---------|
| Only reviewing what's there | Half the job is naming what's missing entirely |
| Burying critical issues in "Summary" | Critical = first, not last |
| Accepting "TBD" / "implementation detail" | Every "TBD" is a risk; call each one out |
| Treating pseudocode as complete | Pseudocode depending on undefined functions is incomplete spec |
| Skipping strengths | Authors need to know what to keep, not just what to fix |
