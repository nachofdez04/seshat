import ssl

import httpx
import pytest

from seshat.core.utils import http_patch


@pytest.fixture(autouse=True)
def _restore_httpx_init():
    """Undo the global monkeypatch so cases don't leak into each other or the wider suite."""
    original = httpx.Client.__init__
    http_patch._ssl_verification_disabled = False
    try:
        yield
    finally:
        httpx.Client.__init__ = original
        http_patch._ssl_verification_disabled = False


class TestDisableHttpxSslVerification:
    def test_client_defaults_to_verify_false_after_patch(self):
        http_patch.disable_httpx_ssl_verification()
        client = httpx.Client()
        ctx = client._transport._pool._ssl_context
        assert ctx.verify_mode == ssl.CERT_NONE  # verification off
        assert ctx.check_hostname is False

    def test_explicit_verify_true_is_still_honoured(self):
        http_patch.disable_httpx_ssl_verification()
        client = httpx.Client(verify=True)
        ctx = client._transport._pool._ssl_context
        assert ctx.verify_mode == ssl.CERT_REQUIRED  # explicit verify=True wins
        assert ctx.check_hostname is True

    def test_is_idempotent(self):
        http_patch.disable_httpx_ssl_verification()
        patched_once = httpx.Client.__init__
        http_patch.disable_httpx_ssl_verification()
        assert httpx.Client.__init__ is patched_once  # second call does not re-wrap
