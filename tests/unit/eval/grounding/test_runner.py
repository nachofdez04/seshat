from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from seshat.agents.grounding import GroundingResult
from seshat.eval.grounding.corpus_loader import GroundingCorpusExample, GroundingCorpusNode
from seshat.eval.grounding.runner import _aggregate_metrics, _build_breakdown, _build_dataframe

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_node(
    title: str = "A node",
    description: str = "Node description.",
    quote: str = "source quote",
    expected_supported: bool = True,
) -> GroundingCorpusNode:
    return GroundingCorpusNode(
        title=title,
        description=description,
        quote=quote,
        expected_supported=expected_supported,
    )


def _make_example(
    corpus_id: str = "ex_001",
    nodes: list[GroundingCorpusNode] | None = None,
    tags: dict | None = None,
) -> GroundingCorpusExample:
    return GroundingCorpusExample(
        corpus_id=corpus_id,
        description="A test example.",
        transcript=None,
        nodes=nodes or [_make_node()],
        tags=tags or {},
    )


def _make_eval_result(result_df: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(result_df=result_df)


# ── _build_dataframe ──────────────────────────────────────────────────────────


def test_build_dataframe_one_row_per_node_not_per_example():
    examples = [
        _make_example("ex1", nodes=[_make_node("N1"), _make_node("N2"), _make_node("N3")]),
        _make_example("ex2", nodes=[_make_node("N4")]),
    ]
    df = _build_dataframe(examples)
    assert len(df) == 4


def test_build_dataframe_has_required_columns():
    df = _build_dataframe([_make_example()])
    assert set(df.columns) == {"inputs", "expectations", "tags"}


def test_build_dataframe_inputs_contains_required_keys():
    node = _make_node(title="My title", description="My desc", quote="My quote")
    ex = _make_example("abc", nodes=[node])
    df = _build_dataframe([ex])
    inputs = df.iloc[0]["inputs"]
    assert inputs["corpus_id"] == "abc"
    assert inputs["node_index"] == 0
    assert inputs["_title"] == "My title"
    assert inputs["_description"] == "My desc"
    assert inputs["_quote"] == "My quote"


def test_build_dataframe_node_index_increments_per_example():
    nodes = [_make_node(f"N{i}") for i in range(3)]
    ex = _make_example("ex1", nodes=nodes)
    df = _build_dataframe([ex])
    indices = [row["node_index"] for row in df["inputs"]]
    assert indices == [0, 1, 2]


def test_build_dataframe_node_index_resets_across_examples():
    ex1 = _make_example("ex1", nodes=[_make_node("A"), _make_node("B")])
    ex2 = _make_example("ex2", nodes=[_make_node("C")])
    df = _build_dataframe([ex1, ex2])
    all_indices = [row["node_index"] for row in df["inputs"]]
    assert all_indices == [0, 1, 0]


def test_build_dataframe_expectations_contain_expected_supported():
    node = _make_node(expected_supported=False)
    df = _build_dataframe([_make_example(nodes=[node])])
    assert df.iloc[0]["expectations"]["expected_supported"] is False


def test_build_dataframe_tags_prefixed_with_corpus():
    ex = _make_example(tags={"source": "zoom", "difficulty": "easy"})
    df = _build_dataframe([ex])
    tags = df.iloc[0]["tags"]
    assert tags["corpus.source"] == "zoom"
    assert tags["corpus.difficulty"] == "easy"


def test_build_dataframe_tags_values_are_strings():
    ex = _make_example(tags={"count": 5})
    df = _build_dataframe([ex])
    tags = df.iloc[0]["tags"]
    assert tags["corpus.count"] == "5"


def test_build_dataframe_empty_input():
    df = _build_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


# ── _aggregate_metrics ────────────────────────────────────────────────────────


def _rows_with_counts(**kwargs) -> pd.DataFrame:
    """Build a single-row result_df with grounding.{k}/value columns."""
    row = {}
    for k, v in kwargs.items():
        row[f"grounding.{k}/value"] = float(v)
    return pd.DataFrame([row])


def test_aggregate_metrics_perfect_precision_and_recall():
    # TP=3, FP=0, FN=0 → precision=1.0, recall=1.0
    df = _rows_with_counts(tp=3, fp=0, fn=0, tn=1)
    result = _aggregate_metrics(_make_eval_result(df))
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)


def test_aggregate_metrics_precision_calculation():
    # TP=2, FP=2, FN=0 → precision=0.5
    df = _rows_with_counts(tp=2, fp=2, fn=0, tn=0)
    result = _aggregate_metrics(_make_eval_result(df))
    assert result["precision"] == pytest.approx(0.5)


def test_aggregate_metrics_recall_calculation():
    # TP=1, FP=0, FN=3 → recall=0.25
    df = _rows_with_counts(tp=1, fp=0, fn=3, tn=0)
    result = _aggregate_metrics(_make_eval_result(df))
    assert result["recall"] == pytest.approx(0.25)


def test_aggregate_metrics_sums_across_rows():
    # Two rows: row1 has TP=1, FP=1; row2 has TP=1, FP=0
    # → total TP=2, FP=1 → precision=2/3
    df = pd.DataFrame(
        [
            {
                "grounding.tp/value": 1.0,
                "grounding.fp/value": 1.0,
                "grounding.fn/value": 0.0,
                "grounding.tn/value": 0.0,
            },
            {
                "grounding.tp/value": 1.0,
                "grounding.fp/value": 0.0,
                "grounding.fn/value": 0.0,
                "grounding.tn/value": 1.0,
            },
        ]
    )
    result = _aggregate_metrics(_make_eval_result(df))
    assert result["precision"] == pytest.approx(2 / 3)


def test_aggregate_metrics_zero_division_precision_returns_zero():
    # TP=0, FP=0 → precision denominator is 0 → 0.0
    df = _rows_with_counts(tp=0, fp=0, fn=2, tn=0)
    result = _aggregate_metrics(_make_eval_result(df))
    assert result["precision"] == pytest.approx(0.0)


def test_aggregate_metrics_zero_division_recall_returns_zero():
    # TP=0, FN=0 → recall denominator is 0 → 0.0
    df = _rows_with_counts(tp=0, fp=1, fn=0, tn=0)
    result = _aggregate_metrics(_make_eval_result(df))
    assert result["recall"] == pytest.approx(0.0)


def test_aggregate_metrics_all_zeros_returns_zero():
    df = _rows_with_counts(tp=0, fp=0, fn=0, tn=0)
    result = _aggregate_metrics(_make_eval_result(df))
    assert result["precision"] == pytest.approx(0.0)
    assert result["recall"] == pytest.approx(0.0)


def test_aggregate_metrics_returns_precision_and_recall_keys():
    df = _rows_with_counts(tp=1, fp=0, fn=0, tn=0)
    result = _aggregate_metrics(_make_eval_result(df))
    assert set(result.keys()) == {"precision", "recall"}


# ── _build_breakdown ──────────────────────────────────────────────────────────


def test_build_breakdown_keys_match_corpus_ids():
    ex = _make_example("example_42")
    result_cache = {("example_42", 0): GroundingResult(supported=True, rationale="Looks good.")}
    breakdown = _build_breakdown([ex], result_cache)
    assert "example_42" in breakdown


def test_build_breakdown_includes_tags():
    ex = _make_example("ex1", tags={"source": "slack"})
    result_cache = {("ex1", 0): GroundingResult(supported=True)}
    breakdown = _build_breakdown([ex], result_cache)
    assert breakdown["ex1"]["tags"] == {"source": "slack"}


def test_build_breakdown_node_fields_from_cache():
    node = _make_node(title="Deploy Redis", expected_supported=True)
    ex = _make_example("ex1", nodes=[node])
    result_cache = {("ex1", 0): GroundingResult(supported=False, rationale="Not grounded.")}
    breakdown = _build_breakdown([ex], result_cache)
    node_out = breakdown["ex1"]["nodes"][0]
    assert node_out["title"] == "Deploy Redis"
    assert node_out["expected_supported"] is True
    assert node_out["predicted_supported"] is False
    assert node_out["rationale"] == "Not grounded."


def test_build_breakdown_missing_cache_entry_yields_none_values():
    node = _make_node(title="Missing node")
    ex = _make_example("ex1", nodes=[node])
    # Provide empty cache — no entry for ("ex1", 0)
    breakdown = _build_breakdown([ex], {})
    node_out = breakdown["ex1"]["nodes"][0]
    assert node_out["predicted_supported"] is None
    assert node_out["rationale"] is None


def test_build_breakdown_one_entry_per_node():
    nodes = [_make_node(f"Node {i}") for i in range(3)]
    ex = _make_example("ex1", nodes=nodes)
    cache = {("ex1", i): GroundingResult(supported=True) for i in range(3)}
    breakdown = _build_breakdown([ex], cache)
    assert len(breakdown["ex1"]["nodes"]) == 3


def test_build_breakdown_multiple_examples():
    ex1 = _make_example("alpha", nodes=[_make_node("A")])
    ex2 = _make_example("beta", nodes=[_make_node("B"), _make_node("C")])
    cache = {
        ("alpha", 0): GroundingResult(supported=True),
        ("beta", 0): GroundingResult(supported=False),
        ("beta", 1): GroundingResult(supported=True),
    }
    breakdown = _build_breakdown([ex1, ex2], cache)
    assert set(breakdown.keys()) == {"alpha", "beta"}
    assert len(breakdown["beta"]["nodes"]) == 2
