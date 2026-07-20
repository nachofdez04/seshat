from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import mlflow
import mlflow.genai
import mlflow.tracing

from seshat.core.utils.hashing import fingerprint
from seshat.eval.corpus_tags import CorpusTagFilter, corpus_tag_summary

if TYPE_CHECKING:
    from collections.abc import Callable

    from mlflow.entities.span import LiveSpan
    from mlflow.tracking.fluent import ActiveModel

    from seshat.core.config.settings import ReflectiveLLMConfig, VectorIndexConfig, _LLMConfig


class Fingerprintable(Protocol):
    def fingerprint(self) -> str: ...
    def prompt_texts(self) -> dict[str, str]: ...


def log_eval_model(
    model_name: str,
    inference_component: Fingerprintable,
    llm_config: _LLMConfig,
    self_review_config: ReflectiveLLMConfig | None = None,
) -> str:
    """Create (or reuse) a versioned MLflow LoggedModel for an LLM-backed agent and return its model_id.

    The model name embeds a fingerprint of its params so each unique config gets its own
    LoggedModel. Re-runs with the same config reuse the existing model without re-logging;
    a changed prompt or LLM config creates a new model automatically.
    The returned model_id can be passed directly to mlflow.genai.evaluate().
    If self_review_config is provided, agent.mode and self_review.enabled tags are set on the active run.
    """
    params = {
        "llm.provider": str(llm_config.provider),
        "llm.model": llm_config.model,
        "llm.temperature": str(llm_config.temperature),
        "agent.prompt_fingerprint": inference_component.fingerprint(),
    }
    model_info = _set_active_model_with_params(model_name, params)

    if not model_info.params:
        for prompt_key, prompt_text in inference_component.prompt_texts().items():
            prompt_name = f"{model_name}-{prompt_key}"
            prompt_version = mlflow.genai.register_prompt(name=prompt_name, template=prompt_text)
            mlflow.genai.load_prompt(prompt_version.uri, link_to_model=True)

    if self_review_config is not None:
        mlflow.set_tags(
            {
                "self_review.enabled": str(self_review_config.enabled).lower(),
                "agent.mode": "reflective" if self_review_config.enabled else "shallow",
            }
        )

    return model_info.model_id


def log_retrieval_model(model_name: str, config: VectorIndexConfig) -> str:
    """Create (or reuse) a versioned MLflow LoggedModel for the retrieval pipeline and return its model_id.

    Fingerprint covers the embedding provider, model, and collection name.
    """
    params = {
        "embedding.provider": str(config.embedding_provider),
        "embedding.model": config.embedding_model,
        "embedding.collection": config.collection,
    }
    return _set_active_model_with_params(model_name, params).model_id


def log_eval_run_metadata(
    run_id: str,
    harness: str,
    gate_passed: bool,
    corpus_dir: Path,
    corpus_examples: list,
    breakdown_artifact: dict | None = None,
    tag_filter: CorpusTagFilter | None = None,
    extra_params: dict[str, str] | None = None,
    cache_hits: int | None = None,
    total_predictions: int | None = None,
) -> None:
    """Log standard eval run params, metrics, tags and artifacts to the MLflow run.

    mlflow.set_tags has no run_id param — it targets the currently active run.
    To ensure tags are set on the correct run, this function should be called within the mlflow.start_run context.
    """
    params: dict[str, str] = {
        "corpus.size": str(len(corpus_examples)),
        "corpus.dir": str(corpus_dir),
    }
    if tag_filter:
        params.update({f"corpus.tag_filter.{k}": str(v) for k, v in tag_filter.items()})
    if cache_hits is not None:
        params["cache.hits"] = str(cache_hits)
        params["cache.total"] = str(total_predictions)
    if extra_params:
        params.update(extra_params)
    mlflow.log_params(params, run_id=run_id)

    mlflow.log_metrics({"gate.passed": float(gate_passed)}, run_id=run_id)

    if breakdown_artifact is not None:
        _log_breakdown_artifact(breakdown_artifact, run_id)

    tag_summary = corpus_tag_summary(corpus_examples)
    tags = {"harness": harness, "gate.passed": str(gate_passed).lower(), **tag_summary}
    if cache_hits is not None and total_predictions is not None:
        tags["cached"] = "true" if cache_hits == total_predictions else "false"
    mlflow.set_tags(tags)


def _set_active_model_with_params(model_name: str, params: dict[str, str]) -> ActiveModel:
    versioned_name = f"{model_name}-{fingerprint(json.dumps(params, sort_keys=True))}"
    model_info = mlflow.set_active_model(name=versioned_name)

    if not model_info.params:
        mlflow.log_model_params(params)

    return model_info


def configure_trace_processors(*processors: Callable[[LiveSpan], None]) -> None:
    """Register one or more span processors with MLflow tracing.

    Each processor is a callable that receives a LiveSpan and mutates it in-place.
    Call once before mlflow.genai.evaluate.
    """
    mlflow.tracing.configure(span_processors=list(processors))


def make_input_redactor(
    fields_to_redact: set[str] | None = None,
    fields_to_exclude: set[str] | None = None,
) -> Callable[[LiveSpan], None]:
    """Return a span processor that sanitises trace inputs.

    fields_to_redact: replaced with **[REDACTED]** (e.g. full transcripts)
    fields_to_exclude: removed entirely (e.g. internal lookup keys)
    """
    redact = fields_to_redact or set()
    exclude = fields_to_exclude or set()

    def _sanitise(span: LiveSpan) -> None:
        if not span.inputs:
            return
        updated = {k: "**[REDACTED]**" if k in redact else v for k, v in span.inputs.items() if k not in exclude}
        if updated != span.inputs:
            span.set_inputs(updated)

    return _sanitise


def _log_breakdown_artifact(breakdown: dict, run_id: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        breakdown_path = Path(tmp_dir) / "breakdown.json"
        breakdown_path.write_text(json.dumps(breakdown, indent=2))
        mlflow.log_artifact(str(breakdown_path), artifact_path="eval", run_id=run_id)
