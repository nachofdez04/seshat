from seshat.core.models.enums import ConceptType

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
RETRIEVAL_MRR_AT_5: float = 0.75

GROUNDING_PRECISION: float = 0.85
GROUNDING_RECALL: float = 0.80

GROUPING_GROUP_HIT_RATE: float = 0.80  # gated — partial credit per example
# grouping.exact_match is logged but not gated (too strict for larger examples)

# UPPER BOUND, unlike every other threshold in this file: WER is lower-is-better, so the
# gate passes when the measured value is <= this number. Placeholder pending calibration
# against the first baseline run.
TRANSCRIPTION_WER_MAX: float = 0.25
# transcription.wer_macro (unweighted mean of per-example WER) is logged but not gated —
# the pooled, length-weighted value is the headline metric.
