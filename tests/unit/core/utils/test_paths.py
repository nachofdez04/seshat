from __future__ import annotations

import pytest

from seshat.core.utils.paths import safe_path_segment, safe_relative_subdir


class TestSafePathSegment:
    @pytest.mark.parametrize(
        "value",
        [
            "meeting_summary.md",
            "job-123",
            "meetings",
            "informe final",
        ],
    )
    def test_safe_segments_pass_unchanged(self, value: str):
        assert safe_path_segment(value, "label") == value

    @pytest.mark.parametrize(
        "value",
        [
            "",
            ".",
            "..",
            ".git",
            ".GIT",
            "a/b",
            "a\\b",
            "trailing.",
            "trailing ",
            "CON",
            "con.md",
            "NUL.txt",
            "COM1",
            "LPT9.md",
            "bad\x00name",
            "bad\x1fname",
            "que?stion",
            "sta*r",
            'quo"te',
            "pi|pe",
            "co:lon",
            "less<than",
        ],
    )
    def test_unsafe_segments_raise(self, value: str):
        with pytest.raises(ValueError, match="label"):
            safe_path_segment(value, "label")


class TestSafeRelativeSubdir:
    def test_multi_segment_path_is_normalized(self):
        assert safe_relative_subdir("docs\\meetings", "docs_subdir") == "docs/meetings"

    def test_single_segment_passes(self):
        assert safe_relative_subdir("meetings", "docs_subdir") == "meetings"

    @pytest.mark.parametrize(
        "value",
        [
            "/absolute",
            "\\absolute",
            "C:\\temp",
            "C:/temp",
            "",
            "a/../b",
            "a/.git/b",
            "a/CON/b",
        ],
    )
    def test_unsafe_paths_raise(self, value: str):
        with pytest.raises(ValueError, match="docs_subdir"):
            safe_relative_subdir(value, "docs_subdir")
