"""End-to-end PublishingService tests: real Postgres ops store + a local bare repo as origin."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING
from unittest.mock import patch
from uuid import uuid4

import pytest

from seshat.app.repositories.ops_repository import OpsRepository
from seshat.app.services.publishing import NothingToPublishError, PublishError, PublishingService
from seshat.core.config.settings import GitPublishingConfig, OpsStoreConfig
from seshat.core.models.documents import DocumentKind, DocumentValidationStatus, GeneratedDocument
from seshat.core.models.enums import JobStatus
from seshat.core.utils.hashing import sha256_text
from seshat.infra.ops_store.pg_store import PostgresOpsStore
from tests.integration.conftest import SKIP_IF_NO_POSTGRES

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

pytestmark = [
    pytest.mark.integration,
    SKIP_IF_NO_POSTGRES,
    pytest.mark.skipif(shutil.which("git") is None, reason="git is not available in the environment"),
]

_MEETING_DATE = date(2026, 7, 22)
_PR_URL = "https://github.com/acme/docs/pull/7"


@pytest.fixture
async def ops_repo(pg_test_url: str) -> AsyncGenerator[OpsRepository]:
    store = PostgresOpsStore(OpsStoreConfig(schema_name="ops"), pg_test_url)
    await store.connect()
    yield OpsRepository(store)
    await store.pool.execute("TRUNCATE ops.jobs, ops.generated_documents, ops.publish_results CASCADE")
    await store.close()


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _setup_repos(tmp_path: Path) -> tuple[Path, Path]:
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


async def _seed_job(ops_repo: OpsRepository, job_id: str, status: JobStatus = JobStatus.DONE) -> None:
    submission = json.dumps({"source_type": "text", "metadata": {"meeting_date": _MEETING_DATE.isoformat()}})
    await ops_repo.create_job(
        job_id,
        "user-1",
        "text",
        None,
        datetime.now(UTC),
        _MEETING_DATE,
        submission,
        f"jobs/{_MEETING_DATE}/{job_id}/raw/input.txt",
    )
    if status != JobStatus.PENDING:
        await ops_repo.update_job_status(job_id, status)


async def _seed_approved_document(ops_repo: OpsRepository, job_id: str, content: str = "# Summary\n") -> None:
    now = datetime.now(UTC)
    document = GeneratedDocument(
        id=uuid4(),
        job_id=job_id,
        kind=DocumentKind.MEETING_SUMMARY,
        filename="meeting_summary.md",
        markdown_content=content,
        content_revision=sha256_text(content),
        created_at=now,
        validation_status=DocumentValidationStatus.APPROVED,
        validated_by="rachel",
        validated_at=now,
        approved_revision=sha256_text(content),
    )
    await ops_repo.upsert_document(document)


def _make_service(ops_repo: OpsRepository, target: Path) -> PublishingService:
    return PublishingService(ops_repo, GitPublishingConfig(enabled=True, target_repo_path=target))


class TestPublishJobEndToEnd:
    async def test_happy_path_persists_and_reaches_origin(self, ops_repo: OpsRepository, tmp_path: Path):
        origin, target = _setup_repos(tmp_path)
        await _seed_job(ops_repo, "job-pub-e2e")
        await _seed_approved_document(ops_repo, "job-pub-e2e")
        service = _make_service(ops_repo, target)

        with (
            patch("seshat.app.services.publishing.pr_module.get_open_pr_url", create=True, return_value=None),
            patch("seshat.app.services.publishing.pr_module.create_pr", return_value=_PR_URL),
        ):
            result = await service.publish_job("job-pub-e2e")

        assert _git(origin, "rev-parse", result.branch)
        latest = await service.get_latest("job-pub-e2e")
        assert latest == result
        assert latest.pr_url == _PR_URL

    async def test_second_publish_without_changes_is_benign(self, ops_repo: OpsRepository, tmp_path: Path):
        _, target = _setup_repos(tmp_path)
        await _seed_job(ops_repo, "job-pub-benign")
        await _seed_approved_document(ops_repo, "job-pub-benign")
        service = _make_service(ops_repo, target)

        with (
            patch("seshat.app.services.publishing.pr_module.get_open_pr_url", create=True, return_value=None),
            patch("seshat.app.services.publishing.pr_module.create_pr", return_value=_PR_URL),
        ):
            await service.publish_job("job-pub-benign")
            with pytest.raises(NothingToPublishError):
                await service.publish_job("job-pub-benign")

        # Only the first publish persisted a result.
        count = await ops_repo._store.pool.fetchval(
            "SELECT COUNT(*) FROM ops.publish_results WHERE job_id='job-pub-benign'"
        )
        assert count == 1

    async def test_unapproved_document_blocks_publish(self, ops_repo: OpsRepository, tmp_path: Path):
        _, target = _setup_repos(tmp_path)
        await _seed_job(ops_repo, "job-pub-pending")
        now = datetime.now(UTC)
        document = GeneratedDocument(
            id=uuid4(),
            job_id="job-pub-pending",
            kind=DocumentKind.MEETING_SUMMARY,
            filename="meeting_summary.md",
            markdown_content="# Summary\n",
            content_revision=sha256_text("# Summary\n"),
            created_at=now,
        )
        await ops_repo.upsert_document(document)
        service = _make_service(ops_repo, target)

        with pytest.raises(PublishError, match="No approved documents"):
            await service.publish_job("job-pub-pending")
