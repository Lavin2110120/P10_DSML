"""Opt-in process-wide TLS verification bypass for corporate proxy environments.

This is intentionally insecure: enable only on a trusted network when the
corporate CA cannot be configured. Set DISABLE_SSL_VERIFY=1 (or true).
"""
from __future__ import annotations

import logging
import os
import ssl
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from typing import Any

_LOG = logging.getLogger(__name__)
_ENABLED = os.getenv("DISABLE_SSL_VERIFY", "0").strip().lower() in {"1", "true"}
_APPLIED = False

def disable_ssl_verification() -> bool:
    """Disable certificate checks for common Python HTTP stacks; safe to repeat.

    Returns True when the bypass is active, False when disabled by environment.
    """
    global _APPLIED
    enabled = os.getenv("DISABLE_SSL_VERIFY", "0").strip().lower() in {"1", "true"}
    if not enabled:
        return False
    if _APPLIED:
        return True

    # Python stdlib HTTPS users (including urllib-based clients).
    ssl._create_default_https_context = ssl._create_unverified_context

    # Environment hints consumed by Requests/cURL-compatible clients.
    for name in ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR"):
        os.environ[name] = ""

    # httpx powers Mistral SDK v1, OpenAI, LangChain integrations, and RAGAS.
    try:
        import httpx

        for cls in (httpx.Client, httpx.AsyncClient):
            original = cls.__init__
            if getattr(original, "_rag_ssl_bypass", False):
                continue
            def _init(self: Any, *args: Any, __original=original, **kwargs: Any) -> None:
                kwargs["verify"] = False
                __original(self, *args, **kwargs)
            _init._rag_ssl_bypass = True  # type: ignore[attr-defined]
            cls.__init__ = _init
    except ImportError:
        _LOG.debug("httpx is not installed; skipping its TLS patch")

    # Requests uses verify in both request() and environment merging.
    try:
        import requests

        request = requests.sessions.Session.request
        if not getattr(request, "_rag_ssl_bypass", False):
            def _request(self: Any, method: str, url: str, *args: Any, **kwargs: Any):
                kwargs["verify"] = False
                return request(self, method, url, *args, **kwargs)
            _request._rag_ssl_bypass = True  # type: ignore[attr-defined]
            requests.sessions.Session.request = _request

        merge = requests.sessions.Session.merge_environment_settings
        if not getattr(merge, "_rag_ssl_bypass", False):
            def _merge(self: Any, url: str, proxies: Any, stream: Any, verify: Any, cert: Any):
                return merge(self, url, proxies, stream, False, cert)
            _merge._rag_ssl_bypass = True  # type: ignore[attr-defined]
            requests.sessions.Session.merge_environment_settings = _merge
    except ImportError:
        _LOG.debug("requests is not installed; skipping its TLS patch")

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass

    _APPLIED = True
    _LOG.warning(
        "DISABLE_SSL_VERIFY is enabled: TLS certificate verification is disabled process-wide. "
        "Use only on a trusted corporate network."
    )
    return True


__all__ = ["disable_ssl_verification"]
