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
    ConceptType.DECISION: 0.75,
    ConceptType.OPEN_QUESTION: 0.75,
    ConceptType.RISK: 0.80,
}

RESOLUTION_PRECISION: float = 0.80
RESOLUTION_RECALL: float = 0.80

RETRIEVAL_RECALL_AT_5: float = 0.70
