import asyncio
import os
import socket
import subprocess
import sys

import pytest
from dotenv import load_dotenv

load_dotenv()

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


def _openai_reachable(openai_api_key_env_var: str | None = None) -> bool:
    key = os.environ.get(openai_api_key_env_var or "OPENAI_API_KEY")
    if not key:
        return False

    import httpx

    try:
        response = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=5,
        )
        return response.status_code < 400
    except httpx.RequestError:
        return False


SKIP_IF_NO_POSTGRES = pytest.mark.skipif(
    not _port_open("localhost", _PG_PORT),
    reason="Postgres not reachable — run: docker compose up -d postgres",
)

SKIP_IF_NO_LOCALSTACK = pytest.mark.skipif(
    not _port_open("localhost", _LOCALSTACK_PORT),
    reason="LocalStack not reachable — run: docker compose up -d localstack",
)

SKIP_IF_NO_OPENAI = pytest.mark.skipif(
    not _openai_reachable(),
    reason="OpenAI API not reachable — OPENAI_API_KEY not set or network issue",
)


@pytest.fixture(scope="session")
def event_loop_policy():
    # psycopg async requires SelectorEventLoop; Windows defaults to ProactorEventLoop
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(scope="session")
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

    _run_alembric_migrations(database_url=_PG_TEST_URL)

    yield _PG_TEST_URL

    admin = await asyncpg.connect(_PG_ADMIN_URL)
    await admin.execute(f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='{_PG_TEST_DB}'")
    await admin.execute(f"DROP DATABASE IF EXISTS {_PG_TEST_DB}")
    await admin.close()


def _run_alembric_migrations(database_url: str):
    env = os.environ | {"DATABASE_URL": database_url}
    subprocess.run(["uv", "run", "alembic", "upgrade", "head"], env=env, check=True)


@pytest.fixture(scope="session")
async def localstack_secretsmanager_url():
    """Create a throw-away Secrets Manager in LocalStack, yield the endpoint URL, then delete it.

    Keeps secrets-manager integration tests isolated from the dev secrets.
    Skipped automatically when LocalStack is not reachable.
    """
    return _get_localstack_url()


@pytest.fixture(scope="session")
async def localstack_s3_url():
    """Create a throw-away S3 bucket in LocalStack, yield the endpoint URL, then delete it.

    Keeps blob-store integration tests isolated from the dev bucket.
    Skipped automatically when LocalStack is not reachable.
    """
    endpoint = _get_localstack_url()

    import aioboto3

    session = aioboto3.Session()
    async with session.client("s3", region_name=LOCALSTACK_REGION, endpoint_url=endpoint) as s3:
        await s3.create_bucket(
            Bucket=LOCALSTACK_TEST_BUCKET,
            CreateBucketConfiguration={"LocationConstraint": LOCALSTACK_REGION},
        )

    yield endpoint

    async with session.client("s3", region_name=LOCALSTACK_REGION, endpoint_url=endpoint) as s3:
        paginator = s3.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=LOCALSTACK_TEST_BUCKET):
            for obj in page.get("Contents", []):
                await s3.delete_object(Bucket=LOCALSTACK_TEST_BUCKET, Key=obj["Key"])
        await s3.delete_bucket(Bucket=LOCALSTACK_TEST_BUCKET)


def _get_localstack_url():
    if not _port_open("localhost", _LOCALSTACK_PORT):
        pytest.skip("LocalStack not reachable — run: docker compose up -d localstack")
    return f"http://localhost:{_LOCALSTACK_PORT}"
