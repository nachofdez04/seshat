---
name: genai-review
description: Use when you want to review a genAI spec, implementation plan, or codebase with a panel of expert agents covering software engineering, data engineering, AI/ML, architecture, security, UX, cost, legal, spec quality, and adversarial perspectives. Invoke after brainstorming or after a major implementation session.
---

# genai-review

Convene an expert review panel for a genAI spec, plan, or codebase. Runs in three phases: parallel domain reviewers → devil's advocate → tech lead synthesis.

## Checklist

You MUST create a TodoWrite task for each of these items before starting, and complete them in order:

1. Detect review target
2. Determine model tier
3. Build agent briefs and mode instructions
4. Phase 1 — Dispatch domain reviewers in parallel
5. Phase 2 — Devil's Advocate
6. Phase 3 — Tech Lead synthesis
7. Present final report to user
8. Write review document
9. User reviews written report
10. Commit

## Step 1: Detect the Review Target

Run `git diff HEAD~1 --name-only` and `git status --short` to identify recently changed files. Map them to review candidates in this order:

1. A file under `docs/superpowers/specs/` appears → candidate for spec review.
2. A file under `docs/superpowers/plans/` appears → candidate for plan review.
3. Other changed files or a meaningful `git diff HEAD~1` output → candidate for code review.
4. If multiple candidates exist, ask the user: "I found [list candidates]. Which should I review? (spec / plan / code)"
5. If none exists, ask: "What should I review? Please provide a file path or describe what you want reviewed."

For **plan review**: auto-locate the matching spec by looking in `docs/superpowers/specs/` for a file with the same date prefix or matching topic name (e.g. `2026-04-22-genai-review.md` pairs with `2026-04-22-genai-review-design.md`). If no match is found, ask the user.

Once the target is identified, ask: **"Include Legal/Compliance review? (y/n, default: n)"**

**Minimum artifact check:** Before proceeding, verify the artifact is substantial enough to review:
- For spec/plan review: the file must be at least 50 lines. If shorter, warn: "This file is only N lines — a full panel review may produce low-quality findings. Proceed? (y/n)"
- For code review: `git diff HEAD~1` must produce at least 20 changed lines. Fewer than 20 lines is not "meaningful output" — ask the user to confirm or provide a different range.

## Step 2: Determine Model Tier

**For spec review and plan review:** `DOMAIN_MODEL = haiku`, `DA_MODEL = sonnet`, `TECHLEAD_MODEL = sonnet`.

**For code review:**
- Run `git diff HEAD~1 | wc -l` to count diff lines.
- Show the user: "Diff is N lines. Use Opus for DA and Tech Lead? (y/n, default: y if N > 500)"
- If N > 500: `DA_MODEL = opus`, `TECHLEAD_MODEL = opus`; user can override to Sonnet.
- If N ≤ 500: `DA_MODEL = sonnet`, `TECHLEAD_MODEL = sonnet`; user can override to Opus.
- `DOMAIN_MODEL = sonnet` for code review regardless of diff size.

After either branch, `DOMAIN_MODEL`, `DA_MODEL`, and `TECHLEAD_MODEL` are now set. Use them in all subsequent steps.

## Step 3: Build the Agent Briefs and Mode Instructions

Determine `{REVIEW_MODE_INSTRUCTIONS}` based on the target type. Three separate values are needed: one for domain reviewers, one for the DA, one for the Tech Lead.

**For spec review:**
- Read the spec file in full.
- Ask: "Any key decisions or constraints I should include in the reviewer briefs?" *(This is a separate targeted question from the Legal/Compliance prompt in Step 1 — ask it even if the user already answered that one.)*
- Agent brief = spec content + constraints summary.
- Domain `{REVIEW_MODE_INSTRUCTIONS}` = `"Review this spec for correctness, completeness, and design quality within your domain."`
- DA `{REVIEW_MODE_INSTRUCTIONS}` = `""` (empty — no additional mandates)
- Tech Lead `{REVIEW_MODE_INSTRUCTIONS}` = `""` (empty — no additional output sections)

**For plan review:**
- Read the plan file in full.
- Read the matching spec file in full (auto-located or provided by user).
- Agent brief = plan content + spec content.
- Domain `{REVIEW_MODE_INSTRUCTIONS}` = `"You are reviewing an implementation plan against the spec it claims to implement. Your job is to find gaps — requirements in the spec that are missing, underspecified, or incorrectly handled in the plan — and flag any steps in your domain that are vague, risky, or sequenced in a way that could cause problems. The spec is the source of truth; the plan must fully satisfy it."`
- DA `{REVIEW_MODE_INSTRUCTIONS}` = `"**Additional mandate:** Check whether the plan's task ordering is safe — are there steps that assume earlier steps completed successfully without verifying? Are there missing rollback steps for destructive operations?"`
- Tech Lead `{REVIEW_MODE_INSTRUCTIONS}` = the Spec Coverage Summary template:

```
### Spec Coverage Summary
*Requirements from the spec that are fully covered, partially covered, or missing from the plan.*

| Spec Requirement | Coverage | Notes |
|---|---|---|
| [requirement] | Full / Partial / Missing | [which task addresses it, or why it's missing] |

---
```

**For code review:**
- Use the diff already fetched in Step 2.
- Identify the spec the code implements (look in `docs/superpowers/specs/` for the matching spec, or ask the user).
- **Multi-tier plan check:** Look in `docs/superpowers/plans/` for plan files whose name shares the same topic root as the spec (e.g. same date prefix or keyword). If 2 or more matching plan files are found, this is a **multi-tier review**:
  - Ask: "I found multiple plan files for this spec: [list]. Which tier is currently being reviewed?"
  - Record `CURRENT_PLAN_FILE` = user-selected file; `OTHER_TIER_PLAN_FILES` = the remaining files.
  - Load prior review reports: scan `docs/genai-review/code/` for files matching the spec topic. Record as `PRIOR_TIER_REVIEW_REPORTS`. If none exist, note "None yet."
  - If only one (or no) plan file is found, this is a **single-tier review** — no scope filtering.
- Agent brief = diff + spec content.
- Domain `{REVIEW_MODE_INSTRUCTIONS}` = `"Review this diff for correctness and quality within your domain. The spec is the intended behaviour; flag any divergence."`
- DA `{REVIEW_MODE_INSTRUCTIONS}` = `""` (empty)
- Tech Lead `{REVIEW_MODE_INSTRUCTIONS}`:
  - **Single-tier:** `TECHLEAD_TIER_CONTEXT = ""`, Tech Lead `{REVIEW_MODE_INSTRUCTIONS}` = `""` (no additional output sections)
  - **Multi-tier:** Set two separate values:

    **`TECHLEAD_TIER_CONTEXT`** — raw plan content, passed as `{TIER_PLAN_CONTEXT}` in the Tech Lead prompt:

    ```
    ## Current Tier Plan

    [full content of CURRENT_PLAN_FILE]

    ## Other Tier Plans

    [for each file in OTHER_TIER_PLAN_FILES:
    "### [filename]
    [full content]
    "]

    ## Prior Tier Review Reports

    [for each file in PRIOR_TIER_REVIEW_REPORTS:
    "### [filename]
    [full content]
    "
    — if none, write "None yet."]
    ```

    **Tech Lead `{REVIEW_MODE_INSTRUCTIONS}`** — output-schema additions only, passed as `{REVIEW_MODE_INSTRUCTIONS}` in the Tech Lead prompt:

    ```
    Using the Tier Plan Context above, classify every finding listed in Consensus Findings,
    Domain-Specific Findings, Unresolved DA Conflicts, and Simplicity/Optimism Flags
    with one of these scope tags:
    - **In Scope** — the current tier plan explicitly tasks this work
    - **Deferred (Tier N)** — a different tier plan covers this; cite the tier filename
    - **Error** — not addressed in any tier plan; a gap that applies regardless of tier

    Add the following two sections to your output, after Simplicity/Optimism Flags and
    before Overall Assessment:

    ### Scope Classification

    | Finding (brief) | Reviewer | Scope | Notes |
    |---|---|---|---|
    | [finding] | [role] | In Scope / Deferred (Tier N) / Error | [which task or tier] |

    ---

    ### Action Items
    *Only In Scope and Error findings. Deferred findings are excluded.*

    1. **[Severity]** [finding] — Evidence: [reference]
    ```

## Step 4: Phase 1 — Dispatch Domain Reviewers in Parallel

Read each reviewer prompt from `reviewer-prompts/` in this skill directory. Substitute:
- `{REVIEW_TARGET_DESCRIPTION}` with a one-line description of what is being reviewed (e.g. "The genai-review skill implementation plan, compared against its design spec.")
- `{ARTIFACT_CONTENT}` with the agent brief built in Step 3
- `{REVIEW_MODE_INSTRUCTIONS}` with the domain value from Step 3

Dispatch the following agents **simultaneously** using the Agent tool, each with `model: DOMAIN_MODEL`:

1. Software Engineer — `reviewer-prompts/software-engineer.md`
2. Data Engineer — `reviewer-prompts/data-engineer.md`
3. AI/ML Engineer — `reviewer-prompts/ai-ml-engineer.md`
4. Solutions Architect — `reviewer-prompts/solutions-architect.md`
5. AI System Designer — `reviewer-prompts/ai-system-designer.md`
6. Security Expert — `reviewer-prompts/security-expert.md`
7. Domain SME / Evaluator — `reviewer-prompts/domain-sme.md`
8. UI/UX Designer — `reviewer-prompts/ux-designer.md`
9. FinOps / Cost Analyst — `reviewer-prompts/finops.md`
10. Spec Quality Analyst — `reviewer-prompts/spec-quality-analyst.md` *(only if target type is spec review)*
11. Plan Quality Analyst — `reviewer-prompts/plan-quality-analyst.md` *(only if target type is plan review)*
12. Legal / Compliance — `reviewer-prompts/legal-compliance.md` *(only if user said yes in Step 1)*

Wait for all agents to return before proceeding.

**Evidence format:** Every finding must follow this structure:
```
Finding: [description]
Evidence: <path>:<line_start>-<line_end>
Severity: Critical | Important | Minor
```
This format applies across all review modes. For spec and plan review, `<path>` is the file path from the agent brief. Single-line citations use `<path>:<line>`.

**Evidence filter:** Before passing reports to Phase 2, scan each report. Remove any finding that does not contain an `Evidence:` line in the format above. Note how many findings were dropped per agent.

**Agent failure handling:** If an agent errors, times out, or returns zero valid findings after filtering, record it as `[AGENT ROLE]: no findings (reason: error / timeout / all filtered)` and continue. Do not re-dispatch. Note the gap in the Review Metadata under "Reviewers".

## Step 5: Phase 2 — Devil's Advocate

Read `reviewer-prompts/devils-advocate.md`. Substitute:
- `{PHASE1_REPORTS}` with the filtered Phase 1 reports (one per agent, labelled by role)
- `{ARTIFACT_CONTENT}` with the same artifact brief used in Phase 1
- `{REVIEW_MODE_INSTRUCTIONS}` with the DA value from Step 3

Dispatch the Devil's Advocate agent with `model: DA_MODEL`. Wait for it to return.

Apply the same evidence filter to the DA report.

## Step 6: Phase 3 — Tech Lead Synthesis

Read `tech-lead-prompt.md`. Substitute:
- `{PHASE1_REPORTS}` with the filtered Phase 1 reports
- `{DA_REPORT}` with the filtered DA report
- `{TIER_PLAN_CONTEXT}` with `TECHLEAD_TIER_CONTEXT` from Step 3 (empty string for single-tier and non-code reviews)
- `{REVIEW_MODE_INSTRUCTIONS}` with the Tech Lead value from Step 3

Dispatch the Tech Lead agent with `model: TECHLEAD_MODEL`. Wait for it to return.

**Multi-tier note:** When `{REVIEW_MODE_INSTRUCTIONS}` contains scope-classification instructions (multi-tier code review), the Tech Lead will produce a Scope Classification table and an Action Items section. The Action Items section is the authoritative TODO list for the current tier — it omits Deferred findings. Present this in Step 7 before writing to disk.

## Step 7: Present the Final Report

Output the Tech Lead report in full. Then add:

```
---
## Review Metadata
- Reviewers: [list of agents dispatched]
- Models: domain=[DOMAIN_MODEL], DA=[DA_MODEL], Tech Lead=[TECHLEAD_MODEL]
- Diff size: [N lines] (code review only)
- Spec file: [path] (plan review only)
- Findings dropped (no evidence): [count per agent]
- DA conflicts flagged: [count]
- Review target: [spec/plan file path or git range]
- Current tier plan: [CURRENT_PLAN_FILE, or "n/a (single-tier)"]
- Other tier plans: [list of OTHER_TIER_PLAN_FILES, or "n/a"]
- Prior tier reviews: [list of PRIOR_TIER_REVIEW_REPORTS, or "none"]
```

## Step 8: Write the Review Document

Create the output directory if needed: `mkdir -p docs/genai-review/<type>/`

Write the full report (Tech Lead output + Review Metadata) to:

```
docs/genai-review/<type>/YYYY-MM-DD-<topic>-review.md
```

Where `<type>` is `specs`, `plans`, or `code`, and `<topic>` matches the reviewed artifact's name.

## Step 9: User Review Gate

Present the path and ask:

> "Review written to `<path>`. Please review it and let me know if you want any changes before I commit."

Wait for the user's response. If they request changes, update the file and ask again. Only proceed once the user approves.

## Step 10: Commit

Commit the review document to git with a message like `"Add genai-review: <topic> (<type>)"`.
