"""Regression test for the truststore/boto3 import-order RecursionError.

truststore.inject_into_ssl() must run before botocore.httpsession is imported anywhere in the
process, otherwise botocore's module-level `from urllib3.util.ssl_ import SSLContext` binds the
pre-injection class, and any later `context.options |= ...` call recurses forever through
ssl.SSLContext.options's super() lookup. This can only be exercised in a fresh interpreter, since
it depends on which modules are already imported — hence the subprocess.
"""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration

_IMPORT_BOTOCORE_BEFORE_INJECT = """
import boto3.exceptions  # unrelated import; boto3.client below is what matters
import botocore.httpsession
import truststore
truststore.inject_into_ssl()
import boto3
boto3.client("secretsmanager", region_name="eu-west-1")
print("OK")
"""

_INJECT_BEFORE_IMPORTING_BOTOCORE = """
import truststore
truststore.inject_into_ssl()
import boto3
boto3.client("secretsmanager", region_name="eu-west-1")
print("OK")
"""

# SECRETS__PROVIDER=aws forces get_secrets_resolver down the path that actually triggers the recursion,
# i.e., AWSSecretsResolver -> boto3.client()
_REAL_APP_STARTUP_SEQUENCE = """
import os
os.environ["USE_OS_TRUSTSTORE"] = "true"
os.environ["SECRETS__PROVIDER"] = "aws"
import seshat.app.platform.api.app
from seshat.core.config.settings import get_config
from seshat.core.utils.http_patch import inject_os_truststore
from seshat.infra.secrets.factory import get_secrets_resolver

config = get_config()
if config.use_os_truststore:
    inject_os_truststore()

get_secrets_resolver(config)
print("OK")
"""


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)


class TestTruststoreBoto3ImportOrder:
    def test_injecting_after_botocore_import_recurses(self):
        """Documents the failure mode this test suite guards against."""
        result = _run(_IMPORT_BOTOCORE_BEFORE_INJECT)
        assert result.returncode != 0
        assert "RecursionError" in result.stderr

    def test_injecting_before_botocore_import_succeeds(self):
        result = _run(_INJECT_BEFORE_IMPORTING_BOTOCORE)
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_real_app_startup_sequence_succeeds(self):
        """seshat.app.platform.api.app must inject truststore before its own imports pull in boto3."""
        result = _run(_REAL_APP_STARTUP_SEQUENCE)
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout
