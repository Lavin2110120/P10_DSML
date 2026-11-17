"""Désactivation globale des vérifications SSL (proxy / inspection TLS)."""

import os
import ssl
import warnings


def _patch_ssl_module():
    # urllib / requests / stdlib
    ssl._create_default_https_context = ssl._create_unverified_context
    # httpx, httpcore, aiohttp, etc. passent par create_default_context
    _orig = ssl.create_default_context

    def _unverified(*args, **kwargs):
        ctx = _orig(*args, **kwargs)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    ssl.create_default_context = _unverified


def _patch_httpx():
    """Force verify=False sur tout client httpx créé après le patch."""
    try:
        import httpx
    except ImportError:
        return
    for cls in (httpx.Client, httpx.AsyncClient):
        if getattr(cls, "_ssl_bypass_patched", False):
            continue
        orig_init = cls.__init__

        def patched(self, *args, _orig=orig_init, **kwargs):
            kwargs.setdefault("verify", False)
            _orig(self, *args, **kwargs)

        cls.__init__ = patched
        cls._ssl_bypass_patched = True


def apply_ssl_bypass():
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    warnings.filterwarnings("ignore", category=UserWarning)

    _patch_ssl_module()

    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass

    _patch_httpx()  # <-- le morceau qui manquait pour google-genai

    os.environ["PYTHONHTTPSVERIFY"] = "0"
    os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = ""
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""


def disable_ssl_verification():
    apply_ssl_bypass()


if os.getenv("DISABLE_SSL_VERIFY", "0") == "1":
    apply_ssl_bypass()