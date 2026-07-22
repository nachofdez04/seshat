"""PublishingService tests against real throwaway git repos — subprocess is never mocked."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from contextlib import contextmanager
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from seshat.app.services.job import JobNotFoundError
from seshat.app.services.publishing import (
    NothingToPublishError,
    PublishError,
    PublishGitError,
    PublishingService,
)
from seshat.core.config.settings import GitPublishingConfig
from seshat.core.models.enums import JobStatus
from seshat.core.utils.hashing import sha256_text

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not available in the environment")

_MEETING_DATE = date(2026, 7, 22)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _setup_repos(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare origin with an initial main branch and a configured working clone."""
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(origin))

    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-b", "main")
    _git(seed, "config", "user.email", "test@example.com")
    _git(seed, "config", "user.name", "Test User")
    (seed / "README.md").write_text("# Docs repo\n", encoding="utf-8")
    _git(seed, "add", "README.md")
    _git(seed, "commit", "-m", "Initial commit")
    _git(seed, "remote", "add", "origin", str(origin))
    _git(seed, "push", "-u", "origin", "main")

    target = tmp_path / "target"
    _git(tmp_path, "clone", str(origin), str(target))
    _git(target, "config", "user.email", "test@example.com")
    _git(target, "config", "user.name", "Test User")
    return origin, target


def _doc_row(job_id: str, content: str = "# Summary\n", **overrides) -> dict:
    row = {
        "id": uuid4(),
        "job_id": job_id,
        "kind": "meeting_summary",
        "filename": "meeting_summary.md",
        "markdown_content": content,
        "content_revision": sha256_text(content),
        "created_at": datetime.now(UTC),
        "validation_status": "approved",
        "validation_revision": 1,
        "edited_content": None,
        "rejection_reason": None,
        "validated_by": "rachel",
        "validated_at": datetime.now(UTC),
        "auto_approved": False,
        "approved_revision": sha256_text(content),
    }
    row.update(overrides)
    return row


def _make_ops(
    job_row: dict | None = ...,
    doc_rows: list[dict] | None = None,
) -> MagicMock:
    ops = MagicMock()
    if job_row is ...:
        job_row = {"job_id": "job-1", "status": JobStatus.DONE, "meeting_date": _MEETING_DATE}

    ops.get_job = AsyncMock(return_value=job_row)
    ops.get_documents_for_job = AsyncMock(return_value=doc_rows if doc_rows is not None else [_doc_row("job-1")])
    ops.insert_publish_result = AsyncMock()
    ops.get_latest_publish_result = AsyncMock(return_value=None)
    return ops


def _make_service(target: Path, ops: MagicMock | None = None, **config_overrides) -> PublishingService:
    fields = {"enabled": True, "target_repo_path": target}
    fields.update(config_overrides)
    return PublishingService(ops if ops is not None else _make_ops(), GitPublishingConfig(**fields))


_PR_URL = "https://github.com/acme/docs/pull/7"


@contextmanager
def _patched_create_pr():
    with (
        patch("seshat.app.services.publishing.pr_module.get_open_pr_url", create=True, return_value=None),
        patch("seshat.app.services.publishing.pr_module.create_pr", return_value=_PR_URL) as create_pr,
    ):
        yield create_pr


class TestPublishJobGuards:
    async def test_disabled_raises(self, tmp_path: Path):
        service = _make_service(tmp_path, enabled=False)
        with pytest.raises(PublishError, match="disabled"):
            await service.publish_job("job-1")

    async def test_unconfigured_repo_raises(self):
        service = PublishingService(_make_ops(), GitPublishingConfig(enabled=True))
        with pytest.raises(PublishError, match="target repository"):
            await service.publish_job("job-1")

    async def test_invalid_base_branch_raises(self, tmp_path: Path):
        service = _make_service(tmp_path, base_branch="bad..branch")
        with pytest.raises(PublishError, match="base_branch"):
            await service.publish_job("job-1")

    async def test_unknown_job_raises(self, tmp_path: Path):
        service = _make_service(tmp_path, ops=_make_ops(job_row=None))
        with pytest.raises(JobNotFoundError):
            await service.publish_job("job-1")

    async def test_non_done_job_raises(self, tmp_path: Path):
        ops = _make_ops(job_row={"job_id": "job-1", "status": JobStatus.PENDING, "meeting_date": _MEETING_DATE})
        service = _make_service(tmp_path, ops=ops)
        with pytest.raises(PublishError, match="done jobs"):
            await service.publish_job("job-1")

    async def test_no_documents_raises(self, tmp_path: Path):
        service = _make_service(tmp_path, ops=_make_ops(doc_rows=[]))
        with pytest.raises(PublishError, match="No approved documents"):
            await service.publish_job("job-1")

    async def test_pending_document_is_not_publishable(self, tmp_path: Path):
        rows = [_doc_row("job-1", validation_status="pending", approved_revision=None)]
        service = _make_service(tmp_path, ops=_make_ops(doc_rows=rows))
        with pytest.raises(PublishError, match="No approved documents"):
            await service.publish_job("job-1")

    async def test_stale_approval_is_blocked(self, tmp_path: Path):
        # Approved, but the approval hash no longer matches the effective content.
        rows = [_doc_row("job-1", approved_revision=sha256_text("something else"))]
        service = _make_service(tmp_path, ops=_make_ops(doc_rows=rows))
        with pytest.raises(PublishError, match="No approved documents"):
            await service.publish_job("job-1")

    async def test_non_markdown_filename_raises(self, tmp_path: Path):
        rows = [_doc_row("job-1", filename="summary.txt")]
        service = _make_service(tmp_path, ops=_make_ops(doc_rows=rows))
        with pytest.raises(PublishError, match="Markdown"):
            await service.publish_job("job-1")


class TestPublishJobGitFlow:
    async def test_happy_path_commits_pushes_and_persists(self, tmp_path: Path):
        origin, target = _setup_repos(tmp_path)
        ops = _make_ops()
        service = _make_service(target, ops=ops)

        with _patched_create_pr():
            result = await service.publish_job("job-1")

        assert result.branch == "seshat/meeting/job-1-2026-07-22"
        assert result.files == ["meetings/job-1/meeting_summary/meeting_summary.md"]
        assert result.pr_url == _PR_URL
        assert result.compare_url == ""
        published = target / "meetings" / "job-1" / "meeting_summary" / "meeting_summary.md"
        assert published.read_text(encoding="utf-8") == "# Summary\n"
        # The commit reached the origin.
        assert _git(origin, "rev-parse", result.branch)
        ops.insert_publish_result.assert_awaited_once_with(result)

    async def test_publishes_edited_content(self, tmp_path: Path):
        _, target = _setup_repos(tmp_path)
        edited = "# Edited summary\n"
        rows = [
            _doc_row("job-1", validation_status="edited", edited_content=edited, approved_revision=sha256_text(edited))
        ]
        service = _make_service(target, ops=_make_ops(doc_rows=rows))

        with _patched_create_pr():
            await service.publish_job("job-1")

        published = target / "meetings" / "job-1" / "meeting_summary" / "meeting_summary.md"
        assert published.read_text(encoding="utf-8") == edited

    async def test_second_publish_without_changes_is_benign(self, tmp_path: Path):
        _, target = _setup_repos(tmp_path)
        service = _make_service(target)

        with _patched_create_pr():
            await service.publish_job("job-1")
            with pytest.raises(NothingToPublishError):
                await service.publish_job("job-1")

        # The working tree returned to the base branch after the benign outcome.
        assert _git(target, "branch", "--show-current") == "main"

    async def test_republish_after_content_change_updates_branch(self, tmp_path: Path):
        origin, target = _setup_repos(tmp_path)
        service = _make_service(target)

        with _patched_create_pr():
            await service.publish_job("job-1")

        changed = "# Summary v2\n"
        service._ops.get_documents_for_job = AsyncMock(return_value=[_doc_row("job-1", content=changed)])
        with _patched_create_pr():
            result = await service.publish_job("job-1")

        assert _git(origin, "rev-parse", result.branch)
        published = target / "meetings" / "job-1" / "meeting_summary" / "meeting_summary.md"
        assert published.read_text(encoding="utf-8") == changed

    async def test_open_pr_is_reused_without_creating_another(self, tmp_path: Path):
        _, target = _setup_repos(tmp_path)
        service = _make_service(target)

        with (
            patch(
                "seshat.app.services.publishing.pr_module.get_open_pr_url",
                create=True,
                return_value=_PR_URL,
            ),
            patch(
                "seshat.app.services.publishing.pr_module.create_pr",
                side_effect=AssertionError("must not create a duplicate PR"),
            ),
        ):
            result = await service.publish_job("job-1")

        assert result.pr_url == _PR_URL
        assert result.compare_url == ""

    async def test_resume_after_failed_push(self, tmp_path: Path):
        origin, target = _setup_repos(tmp_path)
        service = _make_service(target)

        # A pre-receive hook that rejects every push simulates a remote without permissions.
        hook = origin / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8", newline="\n")
        hook.chmod(0o755)

        with _patched_create_pr(), pytest.raises(PublishGitError):
            await service.publish_job("job-1")

        # The local commit exists even though the push was rejected.
        local_sha = _git(target, "rev-parse", "seshat/meeting/job-1-2026-07-22")
        assert local_sha != _git(target, "rev-parse", "main")

        hook.unlink()
        with _patched_create_pr():
            result = await service.publish_job("job-1")

        # The second publish resumed with the pending commit instead of raising.
        assert local_sha.startswith(result.commit_sha)
        assert _git(origin, "rev-parse", result.branch) == local_sha

    async def test_dirty_target_repo_fails_with_git_error(self, tmp_path: Path):
        _, target = _setup_repos(tmp_path)
        (target / "untracked.md").write_text("manual work", encoding="utf-8")
        service = _make_service(target)

        with _patched_create_pr(), pytest.raises(PublishGitError, match="uncommitted"):
            await service.publish_job("job-1")

    async def test_pr_failure_keeps_publish_and_builds_compare_url(self, tmp_path: Path):
        _, target = _setup_repos(tmp_path)
        ops = _make_ops()
        service = _make_service(target, ops=ops)

        from seshat.infra.git_publisher.pr import PrCreationError

        with (
            patch("seshat.app.services.publishing.pr_module.create_pr", side_effect=PrCreationError("gh missing")),
            patch(
                "seshat.app.services.publishing.git_ops.get_remote_url",
                return_value="https://github.com/acme/docs.git",
            ),
        ):
            result = await service.publish_job("job-1")

        assert result.pr_url == ""
        assert result.compare_url == (
            "https://github.com/acme/docs/compare/main...seshat/meeting/job-1-2026-07-22?expand=1"
        )
        ops.insert_publish_result.assert_awaited_once_with(result)

    async def test_concurrent_publishes_are_serialized(self, tmp_path: Path):
        service = _make_service(tmp_path / "unused")
        active = 0
        overlapped = False

        def fake_git_phase(*args, **kwargs) -> str:
            nonlocal active, overlapped
            active += 1
            if active > 1:
                overlapped = True

            time.sleep(0.05)
            active -= 1
            return "abc1234"

        with (
            patch.object(service, "_git_phase", side_effect=fake_git_phase),
            patch.object(service, "_pr_phase", return_value=("", "")),
        ):
            await asyncio.gather(service.publish_job("job-1"), service.publish_job("job-1"))

        assert overlapped is False


class TestGetLatest:
    async def test_returns_none_when_never_published(self, tmp_path: Path):
        service = _make_service(tmp_path)
        assert await service.get_latest("job-1") is None

    async def test_returns_validated_model(self, tmp_path: Path):
        ops = _make_ops()
        ops.get_latest_publish_result = AsyncMock(
            return_value={
                "id": 1,
                "job_id": "job-1",
                "branch": "seshat/meeting/job-1",
                "commit_sha": "abc1234",
                "pr_url": "",
                "compare_url": "",
                "files": ["meetings/job-1/meeting_summary/meeting_summary.md"],
                "published_at": datetime.now(UTC),
            }
        )
        service = _make_service(tmp_path, ops=ops)

        result = await service.get_latest("job-1")

        assert result is not None
        assert result.commit_sha == "abc1234"
