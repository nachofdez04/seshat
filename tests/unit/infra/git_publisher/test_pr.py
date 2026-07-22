from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from seshat.infra.git_publisher import pr as pr_module
from seshat.infra.git_publisher.pr import PrCreationError, build_compare_url, create_pr

if TYPE_CHECKING:
    from pathlib import Path


def _mock_run(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


class TestCreatePr:
    def test_returns_pr_url(self, tmp_path: Path):
        url = "https://github.com/owner/repo/pull/42"
        with patch("subprocess.run", return_value=_mock_run(stdout=f"Some output\n{url}\n")):
            assert create_pr(tmp_path, "branch", "title", "body") == url

    def test_raises_on_nonzero(self, tmp_path: Path):
        with (
            patch("subprocess.run", return_value=_mock_run(returncode=1, stderr="authentication error")),
            pytest.raises(PrCreationError, match="authentication error"),
        ):
            create_pr(tmp_path, "branch", "title", "body")

    def test_raises_when_no_url_in_stdout(self, tmp_path: Path):
        with (
            patch("subprocess.run", return_value=_mock_run(stdout="some non-url output")),
            pytest.raises(PrCreationError, match="valid URL"),
        ):
            create_pr(tmp_path, "branch", "title", "body")

    def test_raises_when_stdout_empty(self, tmp_path: Path):
        with (
            patch("subprocess.run", return_value=_mock_run(returncode=0, stdout="")),
            pytest.raises(PrCreationError, match="valid URL"),
        ):
            create_pr(tmp_path, "branch", "title", "body")

    def test_wraps_missing_executable(self, tmp_path: Path):
        with (
            patch("subprocess.run", side_effect=FileNotFoundError("gh missing")),
            pytest.raises(PrCreationError, match="Could not execute"),
        ):
            create_pr(tmp_path, "branch", "title", "body")

    def test_wraps_timeout(self, tmp_path: Path):
        with (
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("gh", timeout=2)),
            pytest.raises(PrCreationError, match="timed out"),
        ):
            create_pr(tmp_path, "branch", "title", "body", timeout_s=2)

    def test_passes_correct_args(self, tmp_path: Path):
        url = "https://github.com/owner/repo/pull/1"
        with patch("subprocess.run", return_value=_mock_run(stdout=url)) as mock_run:
            create_pr(tmp_path, "my-branch", "My Title", "My Body", base="develop", gh_executable="gh-custom")

        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "gh-custom"
        assert "my-branch" in call_args
        assert "develop" in call_args
        assert "--draft" not in call_args

    def test_draft_flag_added(self, tmp_path: Path):
        url = "https://github.com/owner/repo/pull/1"
        with patch("subprocess.run", return_value=_mock_run(stdout=url)) as mock_run:
            create_pr(tmp_path, "b", "t", "body", draft=True)

        assert "--draft" in mock_run.call_args[0][0]


class TestGetOpenPrUrl:
    def test_returns_open_pr_url(self, tmp_path: Path):
        url = "https://github.com/owner/repo/pull/42"
        with patch("subprocess.run", return_value=_mock_run(stdout=f"{url}\n")) as mock_run:
            result = pr_module.get_open_pr_url(tmp_path, "feature/x", base="develop", gh_executable="gh-custom")

        assert result == url
        assert mock_run.call_args[0][0] == [
            "gh-custom",
            "pr",
            "list",
            "--head",
            "feature/x",
            "--base",
            "develop",
            "--state",
            "open",
            "--json",
            "url",
            "--jq",
            '.[0].url // ""',
        ]

    def test_returns_none_when_no_open_pr_exists(self, tmp_path: Path):
        with patch("subprocess.run", return_value=_mock_run(stdout="\n")):
            assert pr_module.get_open_pr_url(tmp_path, "feature/x") is None


class TestBuildCompareUrl:
    def test_https_remote(self):
        url = build_compare_url("https://github.com/acme/docs.git", "main", "seshat/meeting/x")
        assert url == "https://github.com/acme/docs/compare/main...seshat/meeting/x?expand=1"

    def test_ssh_remote(self):
        url = build_compare_url("git@github.com:acme/docs.git", "main", "feat")
        assert url == "https://github.com/acme/docs/compare/main...feat?expand=1"

    def test_https_without_git_suffix(self):
        url = build_compare_url("https://github.com/acme/docs", "main", "feat")
        assert url == "https://github.com/acme/docs/compare/main...feat?expand=1"

    def test_non_github_returns_none(self):
        assert build_compare_url("https://gitlab.com/acme/docs.git", "main", "feat") is None
