import asyncio
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from aiobotocore.session import get_session
from botocore.exceptions import ClientError

from seshat.app.repositories.blob_repository import BlobRepository
from seshat.core.config.settings import BlobStoreConfig
from seshat.infra.blob_store.s3_store import S3BlobStore
from tests.integration._probes import (
    _anthropic_reachable,
    _assemblyai_reachable,
    _cohere_reachable,
    _openai_direct_reachable,
    _openai_reachable,
    _voyage_reachable,
)
from tests.integration.helpers import make_cheap_llm

_AUDIO_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "audio"

_LOCALSTACK_PORT = int(os.environ.get("LOCALSTACK_PORT", 4566))
LOCALSTACK_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-west-1")
LOCALSTACK_TEST_BUCKET = "seshat-test"

_PG_USER = os.environ.get("POSTGRES_USER", "seshat")
_PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "seshat")
_PG_PORT = int(os.environ.get("POSTGRES_PORT", 5432))
_PG_BASE = f"postgresql://{_PG_USER}:{_PG_PASSWORD}@localhost:{_PG_PORT}"

_PG_DB = os.environ.get("POSTGRES_DB", "seshat")
_PG_ADMIN_URL = f"{_PG_BASE}/{_PG_DB}"

_PG_TEST_DB = "seshat_test"
_PG_TEST_URL = f"{_PG_BASE}/{_PG_TEST_DB}"


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


SKIP_IF_NO_LLM_API = pytest.mark.skipif(
    not _anthropic_reachable() and not _openai_reachable(),
    reason=(
        "No LLM API available — set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "AZURE_OPENAI_* vars, or configure AWS credentials for Bedrock"
    ),
)

SKIP_IF_NO_COHERE_API = pytest.mark.skipif(
    not _cohere_reachable(),
    reason="No Cohere API key — set COHERE_API_KEY",
)

SKIP_IF_NO_VOYAGE_API = pytest.mark.skipif(
    not _voyage_reachable(),
    reason="No Voyage API key — set VOYAGE_API_KEY",
)

SKIP_IF_NO_POSTGRES = pytest.mark.skipif(
    not _port_open("localhost", _PG_PORT),
    reason="Postgres not reachable — run: docker compose up -d postgres",
)

SKIP_IF_NO_LOCALSTACK = pytest.mark.skipif(
    not _port_open("localhost", _LOCALSTACK_PORT),
    reason="LocalStack not reachable — run: docker compose up -d localstack",
)

SKIP_IF_NO_EMBEDDINGS_API = pytest.mark.skipif(
    not _openai_reachable(),
    reason="OpenAI API not reachable — OPENAI_API_KEY not set or network issue",
)

SKIP_IF_NO_ASSEMBLYAI_API = pytest.mark.skipif(
    not _assemblyai_reachable(),
    reason="AssemblyAI API not reachable — ASSEMBLYAI_API_KEY not set or network issue",
)

SKIP_IF_NO_OPENAI_API = pytest.mark.skipif(
    not _openai_reachable(),
    reason="OpenAI API not reachable — OPENAI_API_KEY not set or network issue",
)

SKIP_IF_NO_OPENAI_TRANSCRIPTION_API = pytest.mark.skipif(
    not _openai_direct_reachable(),
    reason="OpenAI transcription API not reachable — OPENAI_API_KEY not set or api.openai.com unreachable",
)


@pytest_asyncio.fixture(loop_scope="function")
async def vector_store(pg_test_url):
    from seshat.core.config.settings import SecretsConfig, SeshatConfig, VectorStoreConfig
    from seshat.core.models.enums import SecretsProvider
    from seshat.infra.vector_store.factory import _build_embeddings
    from seshat.infra.vector_store.pgvector_store import PGVectorStore

    seshat_config = SeshatConfig(secrets=SecretsConfig(provider=SecretsProvider.ENV))
    index = seshat_config.vector_index.model_copy(update={"collection": "test_collection"})
    embeddings = _build_embeddings(index, seshat_config)
    store = PGVectorStore(VectorStoreConfig(), index, embeddings, pg_test_url)
    yield store
    await store._store.adelete_collection()


def pytest_asyncio_loop_factories(config, item):
    # psycopg async requires SelectorEventLoop; Windows defaults to ProactorEventLoop
    if sys.platform == "win32":
        return {"selector": asyncio.WindowsSelectorEventLoopPolicy().new_event_loop}
    return {"default": asyncio.DefaultEventLoopPolicy().new_event_loop}


@pytest.fixture(scope="module")
def cheap_llm():
    return make_cheap_llm()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pg_test_url():
    """Create a throw-away seshat_test database, yield its URL, then drop it.

    Keeps integration tests isolated from real data in the seshat database.
    Skipped automatically when Postgres is not reachable (same check as SKIP_IF_NO_POSTGRES).
    """
    if not _port_open("localhost", _PG_PORT):
        pytest.skip("Postgres not reachable — run: docker compose up -d postgres")

    import asyncpg

    admin = await asyncpg.connect(_PG_ADMIN_URL)
    await admin.execute(f"DROP DATABASE IF EXISTS {_PG_TEST_DB}")
    await admin.execute(f"CREATE DATABASE {_PG_TEST_DB} OWNER {_PG_USER}")
    await admin.close()

    await _init_langchain_tables(_PG_TEST_URL)
    _run_alembric_migrations(database_url=_PG_TEST_URL)

    yield _PG_TEST_URL

    admin = await asyncpg.connect(_PG_ADMIN_URL)
    await admin.execute(f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='{_PG_TEST_DB}'")
    await admin.execute(f"DROP DATABASE IF EXISTS {_PG_TEST_DB}")
    await admin.close()


async def _init_langchain_tables(database_url: str) -> None:
    """Create langchain_pg_* tables before Alembic runs.

    Migration 004 adds a column to langchain_pg_embedding; LangChain creates
    that table lazily, so it must be initialised before Alembic touches it.
    """
    from langchain_core.embeddings.fake import DeterministicFakeEmbedding
    from langchain_postgres import PGVector

    from seshat.core.utils.db import ensure_psycopg_scheme

    pg_url = ensure_psycopg_scheme(database_url)
    store = PGVector(
        embeddings=DeterministicFakeEmbedding(size=1536),
        collection_name="_init",
        connection=pg_url,
        async_mode=True,
    )
    await store.acreate_collection()


def _run_alembric_migrations(database_url: str):
    env = os.environ | {"DATABASE_URL": database_url}
    subprocess.run(["uv", "run", "alembic", "upgrade", "head"], env=env, check=True)


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def localstack_secretsmanager_url():
    """Create a throw-away Secrets Manager in LocalStack, yield the endpoint URL, then delete it.

    Keeps secrets-manager integration tests isolated from the dev secrets.
    Skipped automatically when LocalStack is not reachable.
    """
    return _get_localstack_url()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def localstack_s3_url():
    """Create a throw-away S3 bucket in LocalStack, yield the endpoint URL, then delete it.

    Keeps blob-store integration tests isolated from the dev bucket.
    Skipped automatically when LocalStack is not reachable.
    """
    endpoint = _get_localstack_url()

    session = get_session()
    async with session.create_client("s3", region_name=LOCALSTACK_REGION, endpoint_url=endpoint) as s3:
        try:
            await s3.create_bucket(
                Bucket=LOCALSTACK_TEST_BUCKET,
                CreateBucketConfiguration={"LocationConstraint": LOCALSTACK_REGION},
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
                raise

    yield endpoint

    async with session.create_client("s3", region_name=LOCALSTACK_REGION, endpoint_url=endpoint) as s3:
        paginator = s3.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=LOCALSTACK_TEST_BUCKET):
            for obj in page.get("Contents", []):
                await s3.delete_object(Bucket=LOCALSTACK_TEST_BUCKET, Key=obj["Key"])
        await s3.delete_bucket(Bucket=LOCALSTACK_TEST_BUCKET)


@pytest.fixture(scope="session")
def short_audio_path():
    return _AUDIO_FIXTURES_DIR / "test-audio-short.mp3"


@pytest.fixture(scope="session")
def short_audio_bytes(short_audio_path) -> bytes:
    return short_audio_path.read_bytes()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def blob_store(localstack_s3_url):
    config = BlobStoreConfig(
        bucket=LOCALSTACK_TEST_BUCKET,
        region=LOCALSTACK_REGION,
        endpoint_url=localstack_s3_url,
    )
    store = S3BlobStore(config)
    await store.connect()
    yield BlobRepository(store)
    await store.close()


def _get_localstack_url():
    if not _port_open("localhost", _LOCALSTACK_PORT):
        pytest.skip("LocalStack not reachable — run: docker compose up -d localstack")
    return f"http://localhost:{_LOCALSTACK_PORT}"
