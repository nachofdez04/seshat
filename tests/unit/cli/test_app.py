from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from seshat.cli import _eval_support as support
from seshat.cli.app import app
from seshat.core.config.eval_settings import EvalConfig

# `seshat.cli.__init__` binds the name `app` to the Typer object, shadowing the
# `seshat.cli.app` submodule on attribute access; fetch the real module from sys.modules.
cli_app = sys.modules["seshat.cli.app"]

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


@pytest.fixture
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point EvalConfig's cache dir at a tmp tree with one seeded file per harness."""
    monkeypatch.setattr(EvalConfig, "_cache_dir", tmp_path)
    for harness in ("identification", "resolution", "retrieval", "grounding", "grouping"):
        subdir = tmp_path / harness
        subdir.mkdir()
        (subdir / "seed.json").write_text("{}")

    return tmp_path


class TestClearCacheCommand:
    def test_clears_single_harness_only(self, cache_root: Path) -> None:
        result = runner.invoke(app, ["eval", "clear-cache", "retrieval"])

        assert result.exit_code == 0
        assert list((cache_root / "retrieval").glob("*.json")) == []
        assert (cache_root / "grouping" / "seed.json").exists()

    def test_no_argument_clears_all_harnesses(self, cache_root: Path) -> None:
        result = runner.invoke(app, ["eval", "clear-cache"])

        assert result.exit_code == 0
        for harness in ("identification", "resolution", "retrieval", "grounding", "grouping"):
            assert list((cache_root / harness).glob("*.json")) == []

    def test_unknown_harness_exits_nonzero(self, cache_root: Path) -> None:
        result = runner.invoke(app, ["eval", "clear-cache", "bogus"])

        assert result.exit_code == 1
        assert "Unknown harness" in result.output


class TestHarnessClearCacheFlag:
    def test_flag_clears_cache_before_running(self, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ran = False

        def _fake_run_async(coro: object) -> None:
            nonlocal ran
            ran = True
            coro.close()  # type: ignore[attr-defined]

        monkeypatch.setattr(cli_app, "_run_async", _fake_run_async)

        result = runner.invoke(app, ["eval", "harness", "retrieval", "--clear-cache"])

        assert result.exit_code == 0
        assert list((cache_root / "retrieval").glob("*.json")) == []
        assert ran


class TestHarnessAllFlag:
    @pytest.fixture(autouse=True)
    def _all_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Pin every run_<harness> flag so tests do not depend on a local .env.
        for flag in ("IDENTIFICATION", "RESOLUTION", "RETRIEVAL", "GROUNDING", "GROUPING"):
            monkeypatch.setenv(f"EVAL__RUN_{flag}", "true")

    def test_all_runs_each_enabled_harness(self, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ran: list[str] = []
        monkeypatch.setattr(cli_app, "_run_single_harness", lambda harness, tags: ran.append(harness))

        result = runner.invoke(app, ["eval", "harness", "--all"])

        assert result.exit_code == 0
        assert ran == ["identification", "resolution", "retrieval", "grounding", "grouping"]

    def test_all_respects_disabled_flags(self, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EVAL__RUN_RESOLUTION", "false")
        monkeypatch.setenv("EVAL__RUN_GROUNDING", "false")
        ran: list[str] = []
        monkeypatch.setattr(cli_app, "_run_single_harness", lambda harness, tags: ran.append(harness))

        result = runner.invoke(app, ["eval", "harness", "--all"])

        assert result.exit_code == 0
        assert ran == ["identification", "retrieval", "grouping"]

    def test_all_with_clear_cache_clears_each_run_harness(
        self, cache_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EVAL__RUN_RESOLUTION", "false")
        monkeypatch.setattr(cli_app, "_run_single_harness", lambda harness, tags: None)

        result = runner.invoke(app, ["eval", "harness", "--all", "--clear-cache"])

        assert result.exit_code == 0
        assert list((cache_root / "retrieval").glob("*.json")) == []
        # resolution is disabled, so its cache is left alone
        assert list((cache_root / "resolution").glob("*.json")) != []

    def test_name_and_all_together_errors(self, cache_root: Path) -> None:
        result = runner.invoke(app, ["eval", "harness", "retrieval", "--all"])

        assert result.exit_code == 1
        assert "both" in result.output.lower()

    def test_neither_name_nor_all_errors(self, cache_root: Path) -> None:
        result = runner.invoke(app, ["eval", "harness"])

        assert result.exit_code == 1

    def test_all_with_nothing_enabled_errors(self, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for flag in ("IDENTIFICATION", "RESOLUTION", "RETRIEVAL", "GROUNDING", "GROUPING"):
            monkeypatch.setenv(f"EVAL__RUN_{flag}", "false")

        result = runner.invoke(app, ["eval", "harness", "--all"])

        assert result.exit_code == 1

    def test_all_continues_past_a_failing_harness(self, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ran: list[str] = []

        def _fake(harness: str, tags: object) -> None:
            ran.append(harness)
            if harness == "resolution":
                raise RuntimeError("boom")

        monkeypatch.setattr(cli_app, "_run_single_harness", _fake)

        result = runner.invoke(app, ["eval", "harness", "--all"])

        # every harness is attempted despite resolution raising
        assert ran == ["identification", "resolution", "retrieval", "grounding", "grouping"]
        # a failure makes the overall run exit non-zero
        assert result.exit_code == 1
        # the failed harness is named in the summary
        assert "resolution" in result.output

    def test_all_exits_zero_when_all_succeed(self, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli_app, "_run_single_harness", lambda harness, tags: None)

        result = runner.invoke(app, ["eval", "harness", "--all"])

        assert result.exit_code == 0


class TestHarnessSingleFailHard:
    def test_named_harness_failure_propagates(self, cache_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(harness: str, tags: object) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(cli_app, "_run_single_harness", _boom)

        result = runner.invoke(app, ["eval", "harness", "retrieval"])

        assert result.exit_code != 0


class TestCalibrateClearCacheFlag:
    def test_flag_clears_component_cache_before_running(
        self, cache_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ran = False

        def _fake_run_async(coro: object) -> None:
            nonlocal ran
            ran = True
            coro.close()  # type: ignore[attr-defined]

        monkeypatch.setattr(cli_app, "_run_async", _fake_run_async)

        result = runner.invoke(app, ["eval", "calibrate", "retrieval", "--clear-cache"])

        assert result.exit_code == 0
        assert list((cache_root / "retrieval").glob("*.json")) == []
        assert list((cache_root / "grouping").glob("*.json")) != []
        assert ran


class TestBoundMlflowRetries:
    """A slow/unresponsive MLflow server must fail fast, not retry for ~15 minutes.

    Both the sync HTTP path and the async trace-export path have to be bounded: the
    autolog trace backlog from a large cold harness drains through the async path, which
    defaults to a 500s retry timeout.
    """

    def test_sets_conservative_http_retry_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import os

        for var in ("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "MLFLOW_HTTP_REQUEST_TIMEOUT"):
            monkeypatch.delenv(var, raising=False)

        support.bound_mlflow_retries()

        assert int(os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]) <= 2
        assert float(os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"]) <= 30

    def test_bounds_async_trace_logging_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import os

        monkeypatch.delenv("MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT", raising=False)

        support.bound_mlflow_retries()

        assert float(os.environ["MLFLOW_ASYNC_TRACE_LOGGING_RETRY_TIMEOUT"]) <= 60

    def test_does_not_override_an_explicit_user_setting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import os

        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "9")

        support.bound_mlflow_retries()

        assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "9"


class TestEnsureUtf8Streams:
    """MLflow logs a runner emoji at end_run; a cp1252 Windows console raised
    UnicodeEncodeError. Reconfiguring streams to utf-8 with backslashreplace degrades an
    unencodable char instead of crashing the CLI at shutdown.
    """

    def test_reconfigures_stdout_and_stderr_to_utf8(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        class _Stream:
            def reconfigure(self, **kwargs: object) -> None:
                calls.append(kwargs)

        monkeypatch.setattr(support.sys, "stdout", _Stream())
        monkeypatch.setattr(support.sys, "stderr", _Stream())

        support.ensure_utf8_streams()

        assert len(calls) == 2
        for kw in calls:
            assert kw["encoding"] == "utf-8"
            assert kw["errors"] == "backslashreplace"

    def test_tolerates_streams_without_reconfigure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # older / wrapped streams may lack reconfigure — must not raise
        monkeypatch.setattr(support.sys, "stdout", object())
        monkeypatch.setattr(support.sys, "stderr", object())

        support.ensure_utf8_streams()  # no exception


class TestBootstrapResetsTraceProcessors:
    """In --all, identification registers a global span processor that assumes its own
    node shape; without a reset it leaks onto later harnesses' prediction spans (grounding
    lacks a 'type' key -> hundreds of processor warnings). _bootstrap_eval must clear
    processors once per harness, before predictions.
    """

    def test_bootstrap_clears_trace_processors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import MagicMock

        reset = MagicMock()
        monkeypatch.setattr(support, "configure_trace_processors", reset)
        monkeypatch.setattr(support, "load_dotenv", lambda: None)
        monkeypatch.setattr(support, "_assert_reachable", lambda *a, **k: None)
        monkeypatch.setattr(support, "setup_mlflow", lambda *a, **k: "exp-1")
        monkeypatch.setattr(support, "configure_logging", lambda *a, **k: None)

        support.bootstrap_eval("identification")

        reset.assert_called_once_with()
