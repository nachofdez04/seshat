from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

from seshat.app.platform.observability.mlflow_run_logging import (
    log_identification_failures,
    log_resolution_failures,
    log_token_metrics,
    set_error_tag,
    set_phase_tag,
)
from seshat.core.models.enums import ConceptType
from seshat.core.models.nodes import FailedResolutionSource


def test_log_identification_failures_empty_list_does_not_log():
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.log_metric") as mock_log_metric,
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.set_tag") as mock_set_tag,
    ):
        log_identification_failures([])

    mock_log_metric.assert_not_called()
    mock_set_tag.assert_not_called()


def test_log_identification_failures_logs_count_and_tag():
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.log_metric") as mock_log_metric,
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.set_tag") as mock_set_tag,
    ):
        log_identification_failures([ConceptType.DECISION, ConceptType.RISK])

    mock_log_metric.assert_called_once_with("identification.failed_concept_types", 2)
    mock_set_tag.assert_called_once_with("identification.failed_concept_types", "decision,risk")


def test_log_resolution_failures_empty_list_does_not_log():
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.log_metric") as mock_log_metric,
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.set_tag") as mock_set_tag,
    ):
        log_resolution_failures([])

    mock_log_metric.assert_not_called()
    mock_set_tag.assert_not_called()


def test_log_resolution_failures_logs_count_and_tag():
    node_id = uuid4()
    failed = [FailedResolutionSource(node_id=node_id, concept_type=ConceptType.RISK)]
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.log_metric") as mock_log_metric,
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.set_tag") as mock_set_tag,
    ):
        log_resolution_failures(failed)

    mock_log_metric.assert_called_once_with("resolution.failed_sources", 1)
    mock_set_tag.assert_called_once_with("resolution.failed_sources", str(node_id))


def test_log_token_metrics_no_active_run_does_not_call_log_metrics():
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.active_run", return_value=None),
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.log_metrics") as mock_log,
    ):
        log_token_metrics("my_stage", input_tokens=10, output_tokens=5)

    mock_log.assert_not_called()


def test_log_token_metrics_with_active_run_logs_prefixed_keys():
    fake_run = MagicMock()
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.active_run", return_value=fake_run),
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.log_metrics") as mock_log,
    ):
        log_token_metrics(
            "my_stage",
            input_tokens=10,
            output_tokens=5,
            cache_read_tokens=3,
            cache_creation_tokens=2,
            embedding_input_tokens=7,
        )

    mock_log.assert_called_once_with(
        {
            "usage.my_stage.llm_input": 10.0,
            "usage.my_stage.llm_output": 5.0,
            "usage.my_stage.cache_read_input_tokens": 3.0,
            "usage.my_stage.cache_creation_input_tokens": 2.0,
            "usage.my_stage.embedding_input": 7.0,
        }
    )


def test_log_token_metrics_stage_sanitisation():
    fake_run = MagicMock()
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.active_run", return_value=fake_run),
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.log_metrics") as mock_log,
    ):
        log_token_metrics("step.one two-three", input_tokens=1, output_tokens=1)

    logged = mock_log.call_args[0][0]
    assert "usage.step_one_two_three.llm_input" in logged


def test_log_token_metrics_empty_stage_omits_stage_segment():
    fake_run = MagicMock()
    with (
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.active_run", return_value=fake_run),
        patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.log_metrics") as mock_log,
    ):
        log_token_metrics("", input_tokens=4, output_tokens=2)

    logged = mock_log.call_args[0][0]
    assert "usage.llm_input" in logged
    # No double-dot or spurious stage segment
    for key in logged:
        assert ".." not in key


def test_set_error_tag_truncates_to_250_chars():
    with patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.set_tag") as mock_set_tag:
        set_error_tag(ValueError("x" * 300))

    logged_value = mock_set_tag.call_args[0][1]
    assert mock_set_tag.call_args[0][0] == "error"
    assert len(logged_value) == 250


def test_set_phase_tag_sets_phase_tag():
    with patch("seshat.app.platform.observability.mlflow_run_logging.mlflow.set_tag") as mock_set_tag:
        set_phase_tag("resolution")

    mock_set_tag.assert_called_once_with("phase", "resolution")
