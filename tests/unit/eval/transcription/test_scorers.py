import pytest

from seshat.eval.transcription.scorers import (
    normalize_for_wer,
    pooled_wer,
    word_edit_distance,
    word_error_rate,
)

# ── normalize_for_wer ─────────────────────────────────────────────────────────


def test_normalize_lowercases_and_strips_trailing_punctuation():
    assert normalize_for_wer("Postgres.") == "postgres"


def test_normalize_collapses_whitespace_and_newlines():
    assert normalize_for_wer("  we   need\na  database\t") == "we need a database"


def test_normalize_splits_words_fused_by_punctuation():
    assert normalize_for_wer("search.Postgres") == "search postgres"


def test_normalize_removes_apostrophes_without_splitting_contractions():
    assert normalize_for_wer("I'll") == "ill"
    assert normalize_for_wer("don't") == "dont"


def test_normalize_unifies_unicode_apostrophe_variants():
    assert normalize_for_wer("don\u2019t") == normalize_for_wer("don't")


def test_normalize_preserves_diacritics():
    assert normalize_for_wer("café") == "café"


def test_normalize_keeps_digits():
    assert normalize_for_wer("47 services") == "47 services"


def test_normalize_empty_string():
    assert normalize_for_wer("   ") == ""


# ── word_edit_distance ────────────────────────────────────────────────────────


def test_word_edit_distance_identical_is_zero():
    assert word_edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0


def test_word_edit_distance_substitution():
    assert word_edit_distance(["a", "b", "c"], ["a", "x", "c"]) == 1


def test_word_edit_distance_deletion():
    assert word_edit_distance(["a", "b", "c"], ["a", "c"]) == 1


def test_word_edit_distance_insertion():
    assert word_edit_distance(["a", "c"], ["a", "b", "c"]) == 1


def test_word_edit_distance_empty_reference_is_hypothesis_length():
    assert word_edit_distance([], ["a", "b"]) == 2


def test_word_edit_distance_empty_hypothesis_is_reference_length():
    assert word_edit_distance(["a", "b", "c"], []) == 3


def test_word_edit_distance_both_empty_is_zero():
    assert word_edit_distance([], []) == 0


# ── word_error_rate ───────────────────────────────────────────────────────────


def test_word_error_rate_identical_is_zero():
    assert word_error_rate("we need a database", "we need a database") == pytest.approx(0.0)


def test_word_error_rate_ignores_punctuation_and_case_differences():
    # The whole point of normalizing first: providers differ in punctuation and casing style,
    # and WER must not measure that.
    assert word_error_rate("Postgres.", "postgres") == pytest.approx(0.0)
    assert word_error_rate("We'll use Postgres, not MySQL!", "we'll use postgres not mysql") == pytest.approx(0.0)


def test_word_error_rate_counts_one_substitution():
    assert word_error_rate("we need a database", "we need a warehouse") == pytest.approx(0.25)


def test_word_error_rate_contractions_are_not_expanded():
    # Documented limitation: "I'll" normalizes to one token ("ill"), so matching it against
    # "I will" costs 1 substitution + 1 insertion against a 1-word reference.
    assert word_error_rate("I'll", "I will") == pytest.approx(2.0)


def test_word_error_rate_may_exceed_one_and_is_not_clamped():
    assert word_error_rate("hello", "hello there my old friend") == pytest.approx(4.0)


def test_word_error_rate_empty_reference_and_empty_hypothesis_is_zero():
    assert word_error_rate("", "") == pytest.approx(0.0)


def test_word_error_rate_empty_reference_with_hypothesis_is_one():
    assert word_error_rate("", "some words appeared") == pytest.approx(1.0)


def test_word_error_rate_empty_hypothesis_is_one():
    assert word_error_rate("we need a database", "") == pytest.approx(1.0)


# ── pooled_wer ────────────────────────────────────────────────────────────────


def test_pooled_wer_is_length_weighted_not_macro():
    # Short example: 1 word, 1 error → WER 1.0. Long example: 9 words, 0 errors → WER 0.0.
    # Macro mean would be 0.5; pooled is 1/10.
    long_ref = "one two three four five six seven eight nine"
    pairs = [("alpha", "beta"), (long_ref, long_ref)]
    assert pooled_wer(pairs) == pytest.approx(0.1)


def test_pooled_wer_matches_word_error_rate_for_a_single_pair():
    pairs = [("we need a database", "we need a warehouse")]
    assert pooled_wer(pairs) == pytest.approx(word_error_rate(*pairs[0]))


def test_pooled_wer_perfect_transcription_is_zero():
    pairs = [("first example here", "first example here"), ("second one", "second one")]
    assert pooled_wer(pairs) == pytest.approx(0.0)


def test_pooled_wer_no_pairs_is_zero():
    assert pooled_wer([]) == pytest.approx(0.0)


def test_pooled_wer_only_empty_references_with_hypotheses_is_one():
    assert pooled_wer([("", "unexpected words")]) == pytest.approx(1.0)
