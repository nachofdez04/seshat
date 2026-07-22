"""One-off analysis: same-type vs cross-type precision/recall from the resolution eval cache.

Reprocesses the cached resolution predictions (no LLM calls) to score relations by whether their
source and target ConceptType match (same-type) or differ (cross-type), determined per-relation
from the actual node types rather than the corpus-level same_type/cross_type/mixed tag (which is
too coarse for `mixed` examples). Mirrors the per-example precision/recall then macro-average
methodology already used by seshat.eval.resolution.scorers/runner for the ConceptType breakdown.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "data" / "eval" / "corpora" / "resolution"
CACHE_DIR = PROJECT_ROOT / ".seshat" / "eval_cache" / "resolution"


def load_corpus_examples() -> list[dict]:
    examples = []
    for path in sorted(CORPUS_DIR.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            examples.append(yaml.safe_load(f))
    return examples


def find_cache_file(corpus_id: str) -> Path | None:
    matches = list(CACHE_DIR.glob(f"{corpus_id}_*.json"))
    if not matches:
        return None
    # if multiple agent-hash variants are cached, take the most recently modified
    return max(matches, key=lambda p: p.stat().st_mtime)


def same_cross_group(src_type: str, tgt_type: str) -> str:
    return "same_type" if src_type == tgt_type else "cross_type"


def score_example(
    ex: dict,
    predicted_triples: set[tuple[str, str, str]],
    group_fn,
) -> dict[str, tuple[int, int, int]]:
    """Score one example's triples, bucketed by group_fn(src_type, tgt_type).

    group_fn receives the source and target ConceptType (as strings) of each triple —
    same signature for both the per-source-type replication and the same/cross breakdown,
    so both share this one implementation.
    """
    slug_to_type = {n["id"]: n["type"] for n in ex["source_nodes"] + ex["kb_nodes"]}

    expected_triples = {(r["source"], r["target"], r["rel_type"]) for r in ex["expected_relations"]}

    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    for triple in expected_triples & predicted_triples:
        tp[group_fn(slug_to_type[triple[0]], slug_to_type[triple[1]])] += 1
    for triple in predicted_triples - expected_triples:
        fp[group_fn(slug_to_type[triple[0]], slug_to_type[triple[1]])] += 1
    for triple in expected_triples - predicted_triples:
        fn[group_fn(slug_to_type[triple[0]], slug_to_type[triple[1]])] += 1

    groups = set(tp) | set(fp) | set(fn)
    return {grp: (tp[grp], fp[grp], fn[grp]) for grp in groups}


def load_predicted_triples(ex: dict, corpus_id: str, cache_fp: Path) -> set[tuple[str, str, str]]:
    import json

    cached = json.loads(cache_fp.read_text(encoding="utf-8"))
    slug_map = {n["id"]: uuid5(NAMESPACE_URL, f"{corpus_id}/{n['id']}") for n in ex["source_nodes"] + ex["kb_nodes"]}
    uuid_to_slug = {str(v): k for k, v in slug_map.items()}
    return {
        (
            uuid_to_slug.get(r["source_id"], r["source_id"]),
            uuid_to_slug.get(r["target_id"], r["target_id"]),
            r["rel_type"],
        )
        for r in cached["relationships"]
    }


def run_breakdown(examples: list[dict], group_fn, ordered_groups: list[str]) -> None:
    per_example_scores: dict[str, list[tuple[float, float]]] = defaultdict(list)
    skipped = []

    for ex in examples:
        corpus_id = ex["corpus_id"]
        cache_fp = find_cache_file(corpus_id)
        if cache_fp is None:
            skipped.append(corpus_id)
            continue

        predicted_triples = load_predicted_triples(ex, corpus_id, cache_fp)
        counts = score_example(ex, predicted_triples, group_fn)
        for grp, (tp, fp, fn) in counts.items():
            if tp == 0 and fp == 0 and fn == 0:
                continue
            precision = tp / (tp + fp) if (tp + fp) else (1.0 if not fn else 0.0)
            recall = tp / (tp + fn) if (tp + fn) else 1.0
            per_example_scores[grp].append((precision, recall))

    if skipped:
        print(f"WARNING: no cache file found for {len(skipped)} corpus examples: {skipped}")

    print(f"{'group':<15} {'n_examples':<12} {'precision':<10} {'recall':<10}")
    for grp in ordered_groups:
        scores = per_example_scores[grp]
        n = len(scores)
        mean_p = sum(p for p, _ in scores) / n if n else float("nan")
        mean_r = sum(r for _, r in scores) / n if n else float("nan")
        print(f"{grp:<15} {n:<12} {mean_p:<10.3f} {mean_r:<10.3f}")


_CONCEPT_TYPES = ["decision", "risk", "action_item", "open_question"]


def main() -> None:
    examples = load_corpus_examples()

    print("=== Per source concept type (replicates tab:eval-resolution) ===")
    run_breakdown(
        examples,
        group_fn=lambda src, _tgt: src,
        ordered_groups=_CONCEPT_TYPES,
    )

    print()
    print("=== Same-type vs cross-type (per relation, by actual source/target types) ===")
    run_breakdown(examples, group_fn=same_cross_group, ordered_groups=["same_type", "cross_type"])

    print()
    print("=== Same-type vs cross-type, per source concept type ===")
    ordered = [f"{ct}/{grp}" for ct in _CONCEPT_TYPES for grp in ("same_type", "cross_type")]
    run_breakdown(
        examples,
        group_fn=lambda src, tgt: f"{src}/{same_cross_group(src, tgt)}",
        ordered_groups=ordered,
    )


if __name__ == "__main__":
    main()
