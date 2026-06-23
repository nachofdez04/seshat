from seshat.models.enums import ConceptType

# Targets are intentionally in code, not config — lowering them requires a deliberate,
# reviewable code change.

IDENTIFICATION_PRECISION: dict[ConceptType, float] = {
    ConceptType.ACTION_ITEM: 0.85,
    ConceptType.DECISION: 0.80,
    ConceptType.OPEN_QUESTION: 0.75,
    ConceptType.RISK: 0.75,
}

IDENTIFICATION_RECALL: dict[ConceptType, float] = {
    ConceptType.ACTION_ITEM: 0.85,
    ConceptType.DECISION: 0.80,
    ConceptType.OPEN_QUESTION: 0.75,
    ConceptType.RISK: 0.80,
}

IDENTIFICATION_SPURIOUS_RATE: dict[ConceptType, float] = {
    ConceptType.ACTION_ITEM: 0.10,
    ConceptType.DECISION: 0.10,
    ConceptType.OPEN_QUESTION: 0.15,  # OQ boundaries are genuinely ambiguous
    ConceptType.RISK: 0.10,
}

RESOLUTION_PRECISION: dict[ConceptType, float] = {
    ConceptType.ACTION_ITEM: 0.80,
    ConceptType.DECISION: 0.80,
    ConceptType.OPEN_QUESTION: 0.80,
    ConceptType.RISK: 0.80,
}

RESOLUTION_RECALL: dict[ConceptType, float] = {
    ConceptType.ACTION_ITEM: 0.80,
    ConceptType.DECISION: 0.80,
    ConceptType.OPEN_QUESTION: 0.80,
    ConceptType.RISK: 0.80,
}

RETRIEVAL_RECALL_AT_5: float = 0.70

GROUNDING_PRECISION: float = 0.85
GROUNDING_RECALL: float = 0.80

GROUPING_GROUP_HIT_RATE: float = 0.80  # gated — partial credit per example
# grouping.exact_match is logged but not gated (too strict for larger examples)
