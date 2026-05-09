# genai-review Skill — Design Spec

**Date:** 2026-04-22
**Status:** Approved

## Overview

A skill that convenes an expert review panel for genAI projects at natural inflection points: after brainstorming produces a spec, after planning produces an implementation plan, or after a long implementation session produces code. The panel runs in three sequential phases, combining parallel domain expertise with adversarial challenge and final synthesis.

## Checklist

You MUST create a task for each of these items and complete them in order:

1. **Detect review target** — check git diff and git status for recent candidates
2. **Determine model tier** — based on review type and diff size
3. **Build agent briefs and mode instructions** — read artifact(s), collect constraints
4. **Phase 1 — Dispatch domain reviewers in parallel** — wait for all to return, apply evidence filter
5. **Phase 2 — Devil's Advocate** — wait for return, apply evidence filter
6. **Phase 3 — Tech Lead synthesis** — wait for return
7. **Write review document** — save to `docs/genai-review/<type>/YYYY-MM-DD-<topic>-review.md`
8. **User reviews written report** — wait for approval
9. **Commit** — commit the review document to git

## Invocation

The user invokes `/genai-review`. The skill auto-detects the review target using `git diff HEAD~1 --name-only` and `git status --short`, then maps found files to review candidates in this order:

1. A file under `docs/superpowers/specs/` appears in the diff or status → candidate for spec review
2. A file under `docs/superpowers/plans/` appears in the diff or status → candidate for plan review
3. Other changed files or a meaningful `git diff HEAD~1` → candidate for code review
4. Multiple candidates found → ask user which to review (spec / plan / code)
5. None found → ask user to provide a target explicitly

For **plan review**, the skill auto-locates the matching spec by looking for a spec file with the same date prefix or matching topic name (e.g. `2026-04-22-genai-review.md` pairs with `2026-04-22-genai-review-design.md`). If no match is found, the user is asked.

After confirming the target, the skill asks: **"Include Legal/Compliance review? (y/n, default: n)"**

## Model Tier Selection

Model choice is determined after the target is identified:

**Spec review and plan review:** Haiku for all domain reviewers; Sonnet for Devil's Advocate and Tech Lead.

**Code review:** Run `git diff HEAD~1 | wc -l` to count diff lines. Show the count and ask:
- If diff > 500 lines → default Opus for DA and Tech Lead (user can override to Sonnet)
- If diff ≤ 500 lines → default Sonnet for DA and Tech Lead (user can override to Opus)
- Domain reviewers always use Sonnet for code review

Record three variables for use throughout: `DOMAIN_MODEL`, `DA_MODEL`, `TECHLEAD_MODEL`.

## Review Modes and `{REVIEW_MODE_INSTRUCTIONS}`

Every reviewer prompt, plus the DA and Tech Lead prompts, contains a `{REVIEW_MODE_INSTRUCTIONS}` placeholder. SKILL.md injects a mode-specific paragraph at runtime to orient each agent's lens toward the correct review task.

| Mode | Injected paragraph |
|---|---|
| **Spec** | "Review this spec for correctness, completeness, and design quality within your domain." |
| **Plan** | "You are reviewing an implementation plan against the spec it claims to implement. Your job is to find gaps — requirements in the spec that are missing, underspecified, or incorrectly handled in the plan — and flag any steps in your domain that are vague, risky, or sequenced in a way that could cause problems. The spec is the source of truth; the plan must fully satisfy it." |
| **Code** | "Review this diff for correctness and quality within your domain. The spec is the intended behaviour; flag any divergence." |

The rest of each prompt (lens, focus areas, evidence rule, output format) is unchanged — reviewers apply their existing focus areas through whichever mode lens is injected.

## Agent Briefs

| Target | Context passed to each agent |
|---|---|
| Spec | Full spec doc + constraints summary (key decisions and rationale from brainstorming) |
| Plan | Full plan doc + full matching spec doc |
| Code | Git diff + the spec the diff implements |

For spec review, the skill asks: "Any key decisions or constraints I should include in the reviewer briefs?" before dispatching agents.

## Phases

### Phase 1 — Domain Reviewers (parallel)

Up to 11 domain agents run simultaneously. Each receives the agent brief (no full chat history) with `{REVIEW_MODE_INSTRUCTIONS}` injected.

**Agents:**

| Agent | Primary lens | When dispatched |
|---|---|---|
| Software Engineer | Code quality, async patterns, error handling, testability, maintainability | Always |
| Data Engineer | Ingestion pipelines, data quality, schema evolution, observability | Always |
| AI/ML Engineer | Prompting strategy, RAG vs fine-tune, evaluation framework, hallucination risk, context window management | Always |
| Solutions Architect | Infrastructure design, scalability, component integration, cloud services | Always |
| AI System Designer | Vector DB choices, embedding strategy, retrieval quality, LLM orchestration | Always |
| Security Expert | Prompt injection, data exfiltration via LLM APIs, secrets management, access control | Always |
| Domain SME / Evaluator | Output correctness, domain fitness — are the model outputs actually right? | Always |
| UI/UX Designer | Interaction patterns, latency UX, uncertainty handling in the interface | Always |
| FinOps / Cost Analyst | Token economics, cost per query, caching strategy, model tier selection, scale projections | Always |
| Plan Quality Analyst | Production readiness — implicit requirements, observability, deployment safety, test quality, sequencing | Plan review only |
| Legal / Compliance | GDPR, EU AI Act classification, IP questions, data retention policies | Optional — flagged at invocation |

**Evidence filter (between Phase 1 and Phase 2):** After all domain agents return, scan each report and drop any finding that does not contain an `Evidence:` line. Record how many findings were dropped per agent. Only the surviving findings are passed to the Devil's Advocate.

### Phase 2 — Devil's Advocate

Reads all Phase 1 reports. Attacks from first principles via `{REVIEW_MODE_INSTRUCTIONS}` plus three standing mandates:

- **Simplicity maximalist:** Is this necessary? What's the simplest version that works? Is any layer here premature?
- **Optimism corrector:** Flags over-optimistic assumptions ("this will work at scale", "the model will handle edge cases"). Asks: what if it doesn't?
- **Evidence verifier:** Goes and reads every cited file/line from Phase 1. Calls out any agent that cited incorrectly or out of context.

**Additional mandate for plan review mode:**
- **Ordering safety:** Are there steps that assume earlier steps completed successfully without verifying? Are there missing rollback steps for destructive operations?

**Persona:** Blunt, old-school senior dev. Respectful but pulls no punches. Not impressed by buzzwords.

### Phase 3 — Tech Lead Synthesis

The DA report is also passed through the evidence filter before reaching the Tech Lead. Findings without evidence are dropped.

Synthesizes all Phase 1 reports and the DA's challenge into a final report:

- Domain expertise wins within each agent's lane — conflicting advice from other agents yields to the domain owner
- DA conflicts are **flagged explicitly and left unresolved** for the human to adjudicate (not silently merged)
- Final report is structured, actionable, and prioritized

## Evidence Rule

Every finding by every agent must follow this format:

```
Finding: [description]
Evidence: <path>:<line_start>-<line_end>
Severity: Critical | Important | Minor
```

This format is the same across all three review modes. For spec and plan review, `<path>` is the file path passed in the agent brief and `<line_start>-<line_end>` refers to the line numbers in that file. Single-line citations use `<path>:<line>`.

Findings without a valid `Evidence:` line matching this format are dropped before reaching the Tech Lead. The DA is explicitly permitted to verify cited evidence and challenge agents who cited incorrectly.

## Output and Persistence

After the Tech Lead report is produced:

1. **Write** the report to `docs/genai-review/<type>/YYYY-MM-DD-<topic>-review.md`, where `<type>` is `spec`, `plans`, or `code`.
2. **User Review Gate** — present the path and ask:
   > "Review written to `<path>`. Please review it and let me know if you want any changes before I commit."
   Wait for approval. If changes are requested, update the file and ask again.
3. **Commit** the review document to git.

## Output Structure

```
## Tech Lead Summary

### Consensus Findings
[Findings all or most reviewers agreed on, by severity]

### Domain-Specific Findings
[Findings owned by a single domain expert]

### Unresolved DA Conflicts
[DA pushback that the Tech Lead could not resolve — requires human judgment]

### Simplicity / Optimism Flags
[DA's specific challenges: over-complexity, over-optimism, unnecessary layers]

### Spec Coverage Summary  ← plan review only
| Spec Requirement | Coverage | Notes |
|---|---|---|
| [requirement] | Full / Partial / Missing | [which task addresses it, or why it's missing] |

---
## Review Metadata
- Reviewers: [list of agents dispatched]
- Models: domain=[DOMAIN_MODEL], DA=[DA_MODEL], Tech Lead=[TECHLEAD_MODEL]
- Diff size: [N lines] (code review only)
- Spec file: [path] (plan review only)
- Findings dropped (no evidence): [count per agent]
- DA conflicts flagged: [count]
- Review target: [spec/plan file path or git range]
```

## Constraints and Scope

- **No re-runs in v1.** DA conflicts are flagged, not automatically re-dispatched.
- **Legal/Compliance is optional** — prompted at invocation, default off.
- **Full chat history is never passed to agents.** Context is always a crafted brief.
- **Evidence filter is applied twice:** once after Phase 1 (before DA), once after Phase 2 (before Tech Lead). Dropped finding counts are reported in Review Metadata.
- **`{REVIEW_MODE_INSTRUCTIONS}` is the single extension point** for adding new review targets — no new prompt files needed.
- **Success criterion:** the review document is written to `docs/genai-review/<type>/` and the user approves it before the commit step.

## File Structure

Skill lives in `~/.claude/skills/genai-review/` (personal skills directory).

```
skills/genai-review/
  SKILL.md
  tech-lead-prompt.md
  reviewer-prompts/
    software-engineer.md
    data-engineer.md
    ai-ml-engineer.md
    solutions-architect.md
    ai-system-designer.md
    security-expert.md
    domain-sme.md
    ux-designer.md
    finops.md
    plan-quality-analyst.md
    legal-compliance.md
    devils-advocate.md
```
