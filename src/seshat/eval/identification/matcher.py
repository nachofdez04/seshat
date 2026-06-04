from enum import StrEnum, auto

from pydantic import BaseModel, Field

from seshat.eval.models import IdentificationCorpusNode
from seshat.models.nodes import KBNode

# Minimum score for a (expected, predicted) pair to count as a true positive.
# Pairs below this are rejected: the predicted node becomes spurious (FP) and the expected node
# becomes missed (FN), both hurting precision and recall respectively.
QUOTE_MATCH_THRESHOLD = 0.70


class MatchMethod(StrEnum):
    QUOTE_OVERLAP = auto()
    TITLE_FALLBACK = auto()


class CandidateMatch(BaseModel):
    expected_index: int
    predicted_index: int
    score: float
    method: MatchMethod


class MatchedNode(BaseModel):
    expected: IdentificationCorpusNode
    predicted: KBNode
    match_score: float
    matched_by: MatchMethod


class MatchResult(BaseModel):
    matched: list[MatchedNode] = Field(default_factory=list)
    missed: list[IdentificationCorpusNode] = Field(default_factory=list)
    spurious: list[KBNode] = Field(default_factory=list)


def match_nodes(
    transcript: str,
    expected_corpus_nodes: list[IdentificationCorpusNode],
    predicted_nodes: list[KBNode],
) -> MatchResult:
    """Greedy bipartite match: highest-scoring predicted node claims each expected node first."""
    if not expected_corpus_nodes and not predicted_nodes:
        return MatchResult()

    candidates = _build_candidates(transcript, expected_corpus_nodes, predicted_nodes)
    matched_nodes, claimed_expected_idxs, claimed_predicted_idxs = _greedy_select(
        candidates, expected_corpus_nodes, predicted_nodes
    )
    missed = [node for idx, node in enumerate(expected_corpus_nodes) if idx not in claimed_expected_idxs]
    spurious = [node for idx, node in enumerate(predicted_nodes) if idx not in claimed_predicted_idxs]
    return MatchResult(matched=matched_nodes, missed=missed, spurious=spurious)


def _build_candidates(
    transcript: str,
    expected_corpus_nodes: list[IdentificationCorpusNode],
    predicted_nodes: list[KBNode],
) -> list[CandidateMatch]:
    """Score all same-type (expected, predicted) pairs; return those above threshold, sorted by score descending."""
    candidates: list[CandidateMatch] = []
    for ei, expected_node in enumerate(expected_corpus_nodes):
        for pi, predicted_node in enumerate(predicted_nodes):
            if predicted_node.type != expected_node.type:
                continue

            score, method = _score(expected_node, predicted_node, transcript)
            if score >= QUOTE_MATCH_THRESHOLD:
                candidates.append(CandidateMatch(expected_index=ei, predicted_index=pi, score=score, method=method))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def _score(exp: IdentificationCorpusNode, pred: KBNode, transcript: str) -> tuple[float, MatchMethod]:
    from rapidfuzz import fuzz

    if pred.quote_anchors:
        best = max(
            fuzz.partial_ratio(exp.quote, transcript[a.char_start : a.char_end]) / 100.0 for a in pred.quote_anchors
        )
        return best, MatchMethod.QUOTE_OVERLAP

    title_words = len(exp.title.split()) + len(pred.title.split())
    desc_words = len(exp.description.split()) + len(pred.description.split())
    total = title_words + desc_words or 1
    w_title = title_words / total
    title_ratio = fuzz.token_set_ratio(exp.title, pred.title) / 100.0
    desc_ratio = fuzz.token_set_ratio(exp.description, pred.description) / 100.0
    return w_title * title_ratio + (1 - w_title) * desc_ratio, MatchMethod.TITLE_FALLBACK


def _greedy_select(
    candidates: list[CandidateMatch],
    expected_corpus_nodes: list[IdentificationCorpusNode],
    predicted_nodes: list[KBNode],
) -> tuple[list[MatchedNode], set[int], set[int]]:
    """Assign candidates greedily by score; each node can only be claimed once."""
    claimed_expected_idxs: set[int] = set()
    claimed_predicted_idxs: set[int] = set()
    matched_nodes: list[MatchedNode] = []

    for c in candidates:
        if c.expected_index in claimed_expected_idxs or c.predicted_index in claimed_predicted_idxs:
            continue
        claimed_expected_idxs.add(c.expected_index)
        claimed_predicted_idxs.add(c.predicted_index)
        matched_nodes.append(
            MatchedNode(
                expected=expected_corpus_nodes[c.expected_index],
                predicted=predicted_nodes[c.predicted_index],
                match_score=c.score,
                matched_by=c.method,
            )
        )

    return matched_nodes, claimed_expected_idxs, claimed_predicted_idxs
