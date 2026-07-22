from __future__ import annotations

from datetime import date

import pytest

from seshat.infra.git_publisher.templates import (
    build_branch_name,
    build_commit_message,
    build_pr_body,
    build_pr_title,
    validate_branch_name,
)


class TestValidateBranchName:
    @pytest.mark.parametrize(
        "name",
        [
            "main",
            "seshat/meeting/job-1-2026-07-22",
            "feature/x",
            "release-1.0",
        ],
    )
    def test_valid_names_pass(self, name: str):
        assert validate_branch_name(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "-leading-dash",
            "/leading-slash",
            "trailing-slash/",
            "double//slash",
            "@",
            "dot..dot",
            "at@{brace",
            "trailing.",
            "space in name",
            "tilde~1",
            "caret^",
            "colon:",
            "question?",
            "star*",
            "bracket[",
            "back\\slash",
            "del\x7fchar",
            ".hidden/component",
            "component/.hidden",
            "locked.lock",
            "nested/locked.lock",
        ],
    )
    def test_invalid_names_raise(self, name: str):
        with pytest.raises(ValueError, match="Branch name"):
            validate_branch_name(name)


class TestBuildBranchName:
    def test_combines_prefix_job_and_date(self):
        name = build_branch_name("seshat/meeting", "a1b2c3", date(2026, 7, 22))
        assert name == "seshat/meeting/a1b2c3-2026-07-22"

    def test_without_date(self):
        assert build_branch_name("seshat/meeting", "a1b2c3", None) == "seshat/meeting/a1b2c3"

    def test_slugs_unsafe_job_id(self):
        name = build_branch_name("seshat/meeting", "Diseño · Aprobación", None)
        assert name == "seshat/meeting/diseno-aprobacion"

    def test_prefix_too_long_raises(self):
        with pytest.raises(ValueError, match="too long"):
            build_branch_name("p" * 100, "job", date(2026, 7, 22))

    def test_long_job_id_is_truncated_under_limit(self):
        name = build_branch_name("seshat/meeting", "x" * 200, date(2026, 7, 22))
        assert len(name) <= 100
        validate_branch_name(name)


class TestMessageBuilders:
    def test_commit_message_pluralizes(self):
        assert build_commit_message("job-1", 1).endswith("(1 doc)")
        assert build_commit_message("job-1", 3).endswith("(3 docs)")

    def test_pr_title_mentions_job(self):
        assert "job-1" in build_pr_title("job-1", 2)

    def test_pr_body_lists_files_and_date(self):
        body = build_pr_body("job-1", date(2026, 7, 22), ["meetings/job-1/kind/a.md"])
        assert "`job-1`" in body
        assert "2026-07-22" in body
        assert "- `meetings/job-1/kind/a.md`" in body

    def test_pr_body_without_date_omits_date_line(self):
        body = build_pr_body("job-1", None, ["a.md"])
        assert "Meeting date" not in body
