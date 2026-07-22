from __future__ import annotations

from datetime import UTC, date, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from seshat.app.repositories.blob_repository import BlobRepository
from seshat.app.repositories.ops_repository import OpsRepository
from seshat.app.services.document import DocumentService
from seshat.app.services.job import JobStateError
from seshat.core.config.settings import BlobStoreConfig, OpsStoreConfig
from seshat.core.models.enums import JobStatus
from seshat.core.utils.hashing import sha256_text
from seshat.infra.blob_store.s3_store import S3BlobStore
from seshat.infra.ops_store.pg_store import PostgresOpsStore
from tests.helpers import make_node
from tests.integration.conftest import (
    LOCALSTACK_REGION,
    LOCALSTACK_TEST_BUCKET,
    SKIP_IF_NO_LOCALSTACK,
    SKIP_IF_NO_POSTGRES,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

pytestmark = [pytest.mark.integration, SKIP_IF_NO_POSTGRES, SKIP_IF_NO_LOCALSTACK]

_MEETING_DATE = date(2026, 6, 1)
_TRANSCRIPT = "We agreed to ship v2 in March. Alice will own the rollout plan."


@pytest.fixture
async def ops_repo(pg_test_url: str) -> AsyncGenerator[OpsRepository]:
    store = PostgresOpsStore(OpsStoreConfig(schema_name="ops"), pg_test_url)
    await store.connect()
    yield OpsRepository(store)
    await store.pool.execute("TRUNCATE ops.jobs, ops.generated_documents CASCADE")
    await store.close()


@pytest.fixture
async def blob_repo(localstack_s3_url: str) -> AsyncGenerator[BlobRepository]:
    config = BlobStoreConfig(
        bucket=LOCALSTACK_TEST_BUCKET,
        region=LOCALSTACK_REGION,
        endpoint_url=localstack_s3_url,
    )
    store = S3BlobStore(config)
    await store.connect()
    yield BlobRepository(store)
    await store.close()


def _make_service(ops_repo: OpsRepository, blob_repo: BlobRepository) -> DocumentService:
    node_repo = MagicMock()
    node_repo.paginated_query = AsyncMock(
        return_value=[
            make_node(
                "n1",
                title="Ship v2 in March",
                description="Team agreed to ship v2 in March.",
                quote="We agreed to ship v2 in March.",
                transcript=_TRANSCRIPT,
            )
        ]
    )
    return DocumentService(ops_repo, blob_repo, node_repo)


async def _seed_job(ops_repo: OpsRepository, job_id: str, status: JobStatus) -> None:
    await ops_repo.create_job(
        job_id,
        "user-1",
        "text",
        None,
        datetime.now(UTC),
        _MEETING_DATE,
        '{"source_type": "text", "metadata": {"meeting_date": "2026-06-01"}}',
        f"jobs/2026-06-01/{job_id}/raw/input.txt",
    )
    await ops_repo.update_job_status(job_id, status)


class TestDocumentServiceIntegration:
    async def test_generate_for_done_job_persists_document(self, ops_repo, blob_repo):
        await _seed_job(ops_repo, "job-docsvc-1", JobStatus.DONE)
        await blob_repo.put_raw_transcript(_MEETING_DATE, "job-docsvc-1", _TRANSCRIPT.encode())
        svc = _make_service(ops_repo, blob_repo)

        document = await svc.generate_for_job("job-docsvc-1")

        assert "Ship v2 in March" in document.markdown_content
        assert '[^1]: "We agreed to ship v2 in March."' in document.markdown_content
        assert document.content_revision == sha256_text(document.markdown_content)

        listed = await svc.list_for_job("job-docsvc-1")
        assert [d.id for d in listed] == [document.id]
        fetched = await svc.get(document.id)
        assert fetched == document

    async def test_regeneration_keeps_single_row_and_id(self, ops_repo, blob_repo):
        await _seed_job(ops_repo, "job-docsvc-2", JobStatus.DONE)
        await blob_repo.put_raw_transcript(_MEETING_DATE, "job-docsvc-2", _TRANSCRIPT.encode())
        svc = _make_service(ops_repo, blob_repo)

        first = await svc.generate_for_job("job-docsvc-2")
        second = await svc.generate_for_job("job-docsvc-2")

        assert second.id == first.id
        assert second.content_revision == first.content_revision  # deterministic renderer, unchanged input
        listed = await svc.list_for_job("job-docsvc-2")
        assert len(listed) == 1

    async def test_non_done_job_raises_state_error(self, ops_repo, blob_repo):
        await _seed_job(ops_repo, "job-docsvc-3", JobStatus.AWAITING_REVIEW)
        svc = _make_service(ops_repo, blob_repo)

        with pytest.raises(JobStateError):
            await svc.generate_for_job("job-docsvc-3")
