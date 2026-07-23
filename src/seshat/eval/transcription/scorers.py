from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

import mlflow.genai
from mlflow.entities import Feedback

if TYPE_CHECKING:
    from collections.abc import Sequence

# Apostrophes are deleted rather than turned into a space so contractions stay single tokens
# ("I'll" -> "ill"), and the curly Unicode variants providers disagree on collapse onto the
# straight one. Contractions are NOT expanded: "I'll" vs "I will" still counts as 2 edits.
# Escaped rather than literal so the near-identical glyphs stay distinguishable in source.
_APOSTROPHE_RE = re.compile("['\u2018\u2019\u02bc]")
# Everything else that is neither word nor whitespace becomes a space, so "search.Postgres"
# splits into two tokens instead of fusing into one.
_PUNCT_RE = re.compile(r"[^\w\s]|_")


def normalize_for_wer(text: str) -> str:
    """NFKC -> casefold -> strip punctuation to spaces -> collapse whitespace.

    Diacritics are preserved (casefold, not ASCII-fold): folding them would hide real
    recognition errors if non-English corpora are added later.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _APOSTROPHE_RE.sub("", normalized)
    normalized = _PUNCT_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def word_edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Word-level Levenshtein distance, row-wise DP — O(len(hypothesis)) memory."""
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (0 if ref_word == hyp_word else 1)
            current.append(min(previous[j] + 1, current[j - 1] + 1, substitution))
        previous = current

    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Normalize both sides, then edit operations / reference word count.

    May exceed 1.0 when the hypothesis is much longer than the reference — that is correct
    and deliberately not clamped. An empty reference scores 0.0, or 1.0 if the transcriber
    nonetheless produced words.
    """
    ref_words, hyp_words = _tokenize_pair(reference, hypothesis)
    if not ref_words:
        return 1.0 if hyp_words else 0.0

    return word_edit_distance(ref_words, hyp_words) / len(ref_words)


def pooled_wer(pairs: Sequence[tuple[str, str]]) -> float:
    """Total edit operations over total reference words across all (reference, hypothesis) pairs.

    Length-weighted (micro) average — the gated headline metric. Unlike the macro mean of
    per-example WER, a 10-word example cannot swing the aggregate as much as a 5,000-word one.
    """
    total_edits = 0
    total_ref_words = 0
    for reference, hypothesis in pairs:
        ref_words, hyp_words = _tokenize_pair(reference, hypothesis)
        total_edits += word_edit_distance(ref_words, hyp_words)
        total_ref_words += len(ref_words)

    if not total_ref_words:
        return 1.0 if total_edits else 0.0

    return total_edits / total_ref_words


def _tokenize_pair(reference: str, hypothesis: str) -> tuple[list[str], list[str]]:
    return normalize_for_wer(reference).split(), normalize_for_wer(hypothesis).split()


@mlflow.genai.scorer
def scorer(inputs: dict, outputs: dict, expectations: dict) -> list[Feedback]:
    """Per-example WER for the transcription harness. No LLM calls."""
    return [Feedback(name="transcription.wer", value=word_error_rate(expectations["reference"], outputs["hypothesis"]))]
