"""Process-wide TLS verification bypass, controlled by DISABLE_SSL_VERIFY.

WARNING: Disabling certificate validation is insecure. Use only when explicitly
required on a trusted network; configure the correct CA certificate instead.
"""
from __future__ import annotations

import logging
import os
import ssl
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)
_TRUTHY = {"1", "true", "yes", "y", "on"}
_APPLIED = False


def _load_project_dotenv() -> None:
    """Load the project-root .env independent of the caller's working directory."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)


def _is_enabled() -> bool:
    return os.getenv("DISABLE_SSL_VERIFY", "0").strip().lower() in _TRUTHY


def _remove_proxy_environment() -> None:
    # Do not let environment proxies or NO_PROXY overrides reroute local/API traffic.
    for name in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    ):
        os.environ.pop(name, None)


def disable_ssl_verification() -> bool:
    """Apply TLS bypass to stdlib, httpx, Requests and urllib3; safe to repeat.

    Proxy environment variables are removed irrespective of the TLS setting.
    Returns whether TLS verification bypass is enabled.
    """
    global _APPLIED
    _load_project_dotenv()
    _remove_proxy_environment()
    if not _is_enabled():
        return False
    if _APPLIED:
        return True

    # urllib/stdlib HTTPS calls that use the default HTTPS context.
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["PYTHONHTTPSVERIFY"] = "0"
    # Remove CA overrides which could otherwise change behavior in other stacks.
    for name in ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR"):
        os.environ.pop(name, None)

    # Force all clients constructed after this point to ignore TLS and proxies,
    # including SDK-created clients which do not use our explicitly supplied one.
    try:
        import httpx

        for cls in (httpx.Client, httpx.AsyncClient):
            original = cls.__init__
            if getattr(original, "_project_ssl_bypass", False):
                continue

            def _init(self: Any, *args: Any, __original=original, **kwargs: Any) -> None:
                kwargs["verify"] = False
                kwargs["trust_env"] = False
                __original(self, *args, **kwargs)

            _init._project_ssl_bypass = True  # type: ignore[attr-defined]
            cls.__init__ = _init
    except ImportError:
        _LOG.debug("httpx is not installed; skipping its TLS patch")

    # Requests can apply verification and proxies at both layers.
    try:
        import requests

        session_cls = requests.sessions.Session
        original_request = session_cls.request
        if not getattr(original_request, "_project_ssl_bypass", False):
            def _request(self: Any, method: str, url: str, *args: Any, **kwargs: Any):
                kwargs["verify"] = False
                kwargs["proxies"] = {}
                return original_request(self, method, url, *args, **kwargs)

            _request._project_ssl_bypass = True  # type: ignore[attr-defined]
            session_cls.request = _request

        original_merge = session_cls.merge_environment_settings
        if not getattr(original_merge, "_project_ssl_bypass", False):
            def _merge(self: Any, url: str, proxies: Any, stream: Any, verify: Any, cert: Any):
                return original_merge(self, url, {}, stream, False, cert)

            _merge._project_ssl_bypass = True  # type: ignore[attr-defined]
            session_cls.merge_environment_settings = _merge
    except ImportError:
        _LOG.debug("requests is not installed; skipping its TLS patch")

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        _LOG.debug("urllib3 is not installed; skipping warning suppression")

    _APPLIED = True
    _LOG.warning(
        "DISABLE_SSL_VERIFY is enabled: TLS certificate verification is disabled process-wide. "
        "Use only on a trusted network."
    )
    return True


# Load .env and patch at module import time: indexer imports this module before
# data loaders and SDK clients, so later-created clients inherit the settings.
_load_project_dotenv()
disable_ssl_verification()

__all__ = ["disable_ssl_verification"]
