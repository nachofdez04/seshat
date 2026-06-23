from __future__ import annotations

from unittest.mock import MagicMock, patch

from seshat.observability.mlflow_metrics import log_token_metrics


def test_log_token_metrics_no_active_run_does_not_call_log_metrics():
    with (
        patch("seshat.observability.mlflow_metrics.mlflow.active_run", return_value=None),
        patch("seshat.observability.mlflow_metrics.mlflow.log_metrics") as mock_log,
    ):
        log_token_metrics("my_stage", input_tokens=10, output_tokens=5)

    mock_log.assert_not_called()


def test_log_token_metrics_with_active_run_logs_prefixed_keys():
    fake_run = MagicMock()
    with (
        patch("seshat.observability.mlflow_metrics.mlflow.active_run", return_value=fake_run),
        patch("seshat.observability.mlflow_metrics.mlflow.log_metrics") as mock_log,
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
        patch("seshat.observability.mlflow_metrics.mlflow.active_run", return_value=fake_run),
        patch("seshat.observability.mlflow_metrics.mlflow.log_metrics") as mock_log,
    ):
        log_token_metrics("step.one two-three", input_tokens=1, output_tokens=1)

    logged = mock_log.call_args[0][0]
    assert "usage.step_one_two_three.llm_input" in logged


def test_log_token_metrics_empty_stage_omits_stage_segment():
    fake_run = MagicMock()
    with (
        patch("seshat.observability.mlflow_metrics.mlflow.active_run", return_value=fake_run),
        patch("seshat.observability.mlflow_metrics.mlflow.log_metrics") as mock_log,
    ):
        log_token_metrics("", input_tokens=4, output_tokens=2)

    logged = mock_log.call_args[0][0]
    assert "usage.llm_input" in logged
    # No double-dot or spurious stage segment
    for key in logged:
        assert ".." not in key
