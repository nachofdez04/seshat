# Eval Corpus

Ground-truth fixtures for the evaluation pipelines. Each pipeline has its own sub-directory under `corpus/` and `test_corpus/`.

```
data/eval/
├── corpus/                  # Production corpus used by the eval gate
│   ├── identification/      # Identification eval
│   ├── resolution/          # Resolution eval
│   ├── retrieval/           # Retrieval eval
│   ├── grouping/            # Grouping agent eval
│   └── verification/        # Verification agent eval
└── test_corpus/             # Minimal fixtures used by fast integration tests (not the gate)
    ├── identification/
    ├── resolution/
    ├── retrieval/
    ├── grouping/
    └── verification/
```

`identification_v1/` is a pre-2026-05-30 snapshot kept for reference; it is excluded from all gate and eval runs.

---

## Identification corpus

### Naming convention

```
TIER_NNN_description.yaml
```

| Segment | Values | Meaning |
|---------|--------|---------|
| `TIER` | `happy`, `negative`, `boundary`, `adversarial`, `realistic` | Which test tier (see below) |
| `NNN` | `001`, `002`, … | Sequential within the tier — lower numbers are simpler |
| `description` | kebab-case slug | What makes this fixture distinctive |

To add a new fixture: find the highest `NNN` for that tier and increment.

### Tags (inside each YAML file)

Every identification fixture carries a `tags` block:

```yaml
tags:
  tier: happy-path        # happy-path | negative | boundary | adversarial | realistic
  difficulty: easy        # easy | medium | hard
  focus: [decision, risk] # agent types whose expected output is the diagnostic target
  detail: "..."           # optional: 1–2 word label for the specific rule or case tested
```

`focus` lists the types that are meaningfully asserted. Other types may appear as empty lists or incidental extractions but are not the diagnostic target.

`detail` is free text and should name the specific boundary, rejection criterion, or variant. Omit it for happy-path files where the description is already self-evident.

### Tiers

| Tier prefix | Purpose | Goal |
|------------|---------|------|
| `happy` | Clear, cooperative transcripts — extraction should succeed with high confidence | Baseline precision/recall; failures indicate fundamental extraction problems |
| `negative` | One or more agents must return empty or fewer items — exercises rejection criteria | Drive precision; failures indicate over-extraction on the tested criterion |
| `boundary` | Cross-type interactions where agents most often hallucinate or suppress incorrectly | Validate boundary guidance; failures indicate cross-type confusion |
| `adversarial` | Prompt injection and owner-resolution traps explicitly guarded in the prompts | Verify robustness; failures indicate the guard is not working |
| `realistic` | Longer, naturalistic meeting transcripts with topic drift and multiple speakers | End-to-end validation at production-like complexity |

### Description field vocabulary

Use the vocabulary below when writing the `description` field in a fixture, so failures are easy to categorise:

| Agent | Happy path — what qualifies | Negative — what doesn't | Boundary — where types collide |
|-------|-----------------------------|-------------------------|-------------------------------|
| **decision** | Accepted choice; rejected alternative; rationale stated | Proposal only; preference; unresolved discussion; future revisit with no commitment | Decision vs action implementing it; decision vs answered question; decision vs process note |
| **open_question** | Explicit unresolved question; deferred decision; missing input needed | Rhetorical question; question answered in same exchange; general uncertainty; investigation task only | OQ vs action item to resolve it; OQ vs risk; OQ vs decision deferred to later |
| **risk** | Stated future failure mode with impact or condition | Issue already resolved; general complaint; low-specificity concern; current bug without future uncertainty | Risk vs action mitigating it; risk vs blocker decision; risk vs OQ about unknowns |
| **action_item** | Assigned concrete follow-up with owner; optional due date | Ownerless next step; vague recommendation; current-meeting work only; refused assignment | Action vs decision implementation; action vs OQ investigation; action vs risk mitigation |

---

## Resolution corpus

Fixtures for the resolution agent, which determines relationships between KBNode pairs.

### Naming convention

```
TIER_NNN_description.yaml
```

| Segment | Values | Meaning |
|---------|--------|---------|
| `TIER` | `simple`, `null`, `multi`, `realistic` | Which test tier (see below) |
| `NNN` | `001`, `002`, … | Sequential within the tier — lower numbers are simpler |
| `description` | kebab-case slug | Source type, target type, and relationship or scenario |


### Tiers

| Tier prefix | Node count | Purpose |
|------------|-----------|---------|
| `simple` | 1 source + 1 KB | Single-pair recognition — one obvious expected label (or one obvious null) |
| `null` | 1–2 sources + 1–3 KB | Precision discipline — all-null expected output; thematically adjacent but unlinked nodes |
| `multi` | 1 source + 4–6 KB | Mixed labels — realistic lookup scenario; some targets related, some null |
| `realistic` | 3–5 sources + 5–10 KB | Full meeting context — multiple source and KB nodes, cross-type mixed |

### Relation type coverage targets

Test case counts should track **boundary complexity**, not be uniform. Relationship types that are harder to discriminate warrant more coverage:

| Relation | Target coverage | Rationale |
|----------|----------------|-----------|
| `amends` | High | Hardest boundary: requires same concern domain, source must narrow/qualify without replacing. Easily confused with `supersedes` (when source sounds sweeping) and `null` (when concerns are adjacent but not the same layer). |
| `conflicts_with` | High | Requires both nodes to be currently active and mutually incompatible — easily confused with `supersedes`. |
| `supersedes` | High | Frequently predicted incorrectly in both directions (fires too broadly; misses when source only narrows or contradicts). |
| `blocks` | Medium–high | Multiple source types can block; criterion ("prevents execution as stated") is concrete but cross-type pairing rules add surface area. |
| `depends_on` | Medium | Directional but clear once the dependency direction is established. |
| `mitigates` | Medium | "Mechanistically reduces the failure mode" is precise; spurious predictions are usually transitive-inference errors in realistic fixtures, not boundary confusion. |
| `resolves` | Lower | Cleanest criterion: does the decision fully and directly answer the question? Rarely ambiguous. Cross-type performance is consistently near-perfect. |

---

## Retrieval corpus

Fixtures for the retrieval scorer (recall@5 gated, precision@5 logged). Each fixture seeds a candidate pool into pgvector, runs `NodeRetriever.retrieve()` on a single query node, and checks which candidates appear in the top-5 results.

### Naming convention

```
NNN_description.yaml
```

Flat sequence — no tier prefix. The description should encode the query type, candidate type, and scenario (e.g. `decision_risk_cross_type`, `no_relation_thematically_adjacent`).

### YAML format

```yaml
corpus_id: "<slug>"
description: "<what this fixture tests>"

query_node:
  id: <slug>
  type: <concept_type>
  title: "..."
  description: "..."
  quote: "..."

candidate_nodes:
  - id: <slug>
    type: <concept_type>
    title: "..."
    description: "..."
    quote: "..."
  # ... more candidates

expected_relevant_ids:
  - <slug>   # slugs from candidate_nodes that should appear in the top-5 results
```

### Coverage targets

| Scenario | Purpose |
|----------|---------|
| Same-type, same domain | Baseline: semantically similar nodes of the same type should surface |
| Cross-type, logically related | Validates cross-type retrieval (e.g. a Risk surfacing related Decisions) |
| No relation, thematically adjacent | Precision discipline: nodes on the same broad topic but no logical link should not dominate top-5 |

---

## Grouping corpus

Fixtures for the grouping agent, which clusters extraction nodes that refer to the same underlying item.

### Naming convention

```
TIER_NNN_description.yaml
```

| Segment | Values | Meaning |
|---------|--------|---------|
| `TIER` | `merge`, `mixed`, `singleton`, `realistic` | Which test tier (see below) |
| `NNN` | `001`, `002`, … | Sequential within the tier — lower numbers are simpler |
| `description` | kebab-case slug | What makes this fixture distinctive |

To add a new fixture: find the highest `NNN` for that tier and increment.

### Tiers

| Tier prefix | Purpose |
|------------|---------|
| `merge` | Multiple items that should be grouped together into one cluster |
| `mixed` | Combination of items that merge and items that remain as singletons |
| `singleton` | Items that should each form their own singleton cluster — no merging |
| `realistic` | Longer, naturalistic transcripts testing end-to-end grouping behaviour |

---

## Verification corpus

Fixtures for the verification agent, which checks whether a quoted excerpt supports a stated claim.

### Naming convention

```
TIER_NNN_description.yaml
```

| Segment | Values | Meaning |
|---------|--------|---------|
| `TIER` | `faithful`, `hallucination`, `adversarial` | Which test tier (see below) |
| `NNN` | `001`, `002`, … | Sequential within the tier — lower numbers are simpler |
| `description` | kebab-case slug | What makes this fixture distinctive |

To add a new fixture: find the highest `NNN` for that tier and increment.

### Tiers

| Tier prefix | Expected agent output | Purpose |
|------------|----------------------|---------|
| `faithful` | `supported` | The quote genuinely supports the claim — verifies the agent does not over-reject |
| `hallucination` | `unsupported` | The claim adds facts not present in the quote — verifies the agent catches fabrication |
| `adversarial` | not fooled | Prompt injection or identity manipulation traps — verifies the agent's robustness guards |
