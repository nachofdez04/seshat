"""Shared HTTP client tweaks.

Home of the SSL-verification opt-out used when running behind a corporate proxy
whose CA chain httpx cannot see (e.g. a Windows laptop where the cert store is not
exposed to Python). Gated by config; off by default.
"""

from __future__ import annotations

import httpx

from seshat.core.utils.log import get_logger

logger = get_logger(__name__)

_ssl_verification_disabled = False


def disable_httpx_ssl_verification() -> None:
    """Monkeypatch httpx.Client so it defaults to ``verify=False`` process-wide.

    INSECURE: this disables TLS certificate verification for every httpx.Client
    that does not pass ``verify`` explicitly. Only call it as a deliberate,
    config-gated escape hatch for a trusted network (corporate proxy with an
    un-seen CA); never enable it in production. Callers that pass ``verify=True``
    explicitly are unaffected. Idempotent — calling twice does not re-wrap.
    """
    global _ssl_verification_disabled
    if _ssl_verification_disabled:
        return

    _original_init = httpx.Client.__init__

    def _init_no_verify(self: httpx.Client, *args: object, **kwargs: object) -> None:
        kwargs.setdefault("verify", False)
        _original_init(self, *args, **kwargs)  # type: ignore[arg-type]

    httpx.Client.__init__ = _init_no_verify  # type: ignore[method-assign]
    _ssl_verification_disabled = True
    logger.warning("httpx SSL certificate verification DISABLED process-wide (config opt-in).")
