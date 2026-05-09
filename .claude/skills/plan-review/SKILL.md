---
name: plan-review
description: Use when asked to review, critique, or improve an implementation plan, most likely  produced by the writing-plans skill — focuses on what writing-plans misses: production risk, observability, cross-task consistency, implicit requirements, and compatibility
---

# Plan Review

## Overview

Plans produced by `writing-plans` are already self-reviewed for spec coverage, placeholder content, and type consistency. This skill covers what that self-review doesn't: production and ops risk, observability, architecture coherence across tasks, and implicit requirements the spec never mentioned but production demands.

**On invocation, use TodoWrite to create one todo per checklist section below, then work through each in order.**

## Expected Input

Plans produced by `writing-plans` follow this structure:

```
# [Feature] Implementation Plan
Goal / Architecture / Tech Stack header
---
### Task N: [Component Name]
Files: Create / Modify / Test (exact paths)
- [ ] Step 1: Write the failing test  [code block]
- [ ] Step 2: Run test, verify FAIL    [command + expected output]
- [ ] Step 3: Write minimal impl       [code block]
- [ ] Step 4: Run test, verify PASS    [command + expected output]
- [ ] Step 5: Commit                   [git command]
```

Do **not** flag missing Owner / Review-needed fields — writing-plans doesn't produce those.

## Review Checklist

### 1. Cross-Task Architecture Consistency
These are NOT covered by writing-plans' own self-review:

- Do type signatures, method names, and interfaces defined in early tasks remain consistent through later tasks? (writing-plans checks this, but catch any misses)
- Does the file structure defined in the header actually match the files touched in each task?
- Are there tasks that silently assume a shared abstraction that was never defined?
- Does the order of task commits leave the codebase in a working state at each step, or are there intermediate broken states?

### 2. Production & Ops Risk
Writing-plans focuses on buildability; it does not check deployability:

- Are database migrations or schema changes present? If so:
  - Is the migration reversible (down migration exists)?
  - Does any task deploy new code before the migration runs, or vice versa?
- Are there background jobs, queues, or async workers? Are they handled gracefully during deployment?
- Is there a rollback plan if Task N is deployed and needs to be reverted?
- Are there third-party service dependencies (APIs, SDKs) that could be unavailable?

### 3. Implicit Requirements
Things production needs that specs typically omit:

- Authentication / authorization — does the new code respect existing access control?
- Rate limiting / throttling — if a new endpoint or job is added, is it bounded?
- Data validation at system boundaries (user input, external API responses)
- Error handling for external calls — timeouts, retries, circuit breakers where applicable
- Resource cleanup — files, DB connections, temp objects

### 4. Observability
Almost never present in plans, always needed in production:

- Is there at least one log line per task that would confirm the feature is running in prod?
- Are errors logged with enough context to diagnose without a debugger?
- If a metric or trace is relevant (new endpoint, new job), is it tracked anywhere in the plan?
- How will anyone know the feature is healthy the day after launch?

### 5. Compatibility
- Are existing API contracts or data schemas changed? Are old clients still supported?
- If a field is renamed or removed, is there a migration or deprecation step?
- Are tests updated for behavior that changed, not just new behavior?

### 6. Test Quality
Writing-plans checks for presence of tests; check the quality:

- Do tests assert on product behavior (what the user experiences), or only on implementation details (internal function calls)?
- Are failure paths tested, not just the happy path?
- Are tests independent — can each one run in isolation without depending on state from a previous test?

### 7. Sequencing
- Can the plan be executed linearly as written, or does a later task require something an earlier task didn't produce?
- Is there a visible, working checkpoint after each milestone — a point where you could stop and have something shippable?

## Output Format

```
## Critical Issues
[Broken intermediate states, irreversible migrations deployed in wrong order,
missing auth on new endpoints — things that will cause incidents]

## High-Priority Issues
[No observability, unhandled external failure modes, tests that only cover happy path,
missing rollback plan]

## Medium-Priority Issues
[Minor cross-task inconsistencies, missing edge case validation, weak error messages]

## What's Missing Entirely
[Implicit requirements the plan doesn't address at all]

## Strengths
[What the plan does well — required, not optional]

## Clarifying Questions
[Concrete questions that must be answered before execution begins]
```

## Common Mistakes

| Mistake | Reality |
|---------|---------|
| Flagging missing Owner/AC fields | writing-plans doesn't use that format; don't penalize it |
| Re-checking spec coverage | writing-plans already does this; skip unless you spot a clear gap |
| Reviewing only the code path | Deployment order and migration reversibility cause most incidents |
| Accepting "log as needed" | Logging is a first-class requirement, not an afterthought |
| Skipping strengths | Authors need to know what to keep, not just what to fix |
