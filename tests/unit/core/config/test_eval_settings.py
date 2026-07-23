from __future__ import annotations

import pytest

from seshat.core.config.eval_settings import EvalConfig


class TestCacheDirFor:
    @pytest.mark.parametrize(
        ("harness", "subdir"),
        [
            ("identification", "identification"),
            ("resolution", "resolution"),
            ("retrieval", "retrieval"),
            ("grounding", "grounding"),
            ("grouping", "grouping"),
            ("transcription", "transcription"),
        ],
    )
    def test_maps_each_harness_to_its_cache_subdir(self, harness: str, subdir: str) -> None:
        result = EvalConfig.cache_dir_for(harness)
        assert result == EvalConfig._cache_dir / subdir

    def test_unknown_harness_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="bogus"):
            EvalConfig.cache_dir_for("bogus")


class TestEnabledHarnesses:
    def test_all_enabled_by_default(self) -> None:
        cfg = EvalConfig()
        assert cfg.enabled_harnesses == [
            "identification",
            "resolution",
            "retrieval",
            "grounding",
            "grouping",
            "transcription",
        ]

    def test_disabled_harness_is_excluded(self) -> None:
        cfg = EvalConfig(run_resolution=False, run_grounding=False)
        assert cfg.enabled_harnesses == ["identification", "retrieval", "grouping", "transcription"]

    def test_none_enabled_returns_empty(self) -> None:
        cfg = EvalConfig(
            run_identification=False,
            run_resolution=False,
            run_retrieval=False,
            run_grounding=False,
            run_grouping=False,
            run_transcription=False,
        )
        assert cfg.enabled_harnesses == []
