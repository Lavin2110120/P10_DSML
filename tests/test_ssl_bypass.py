"""Unit tests for the opt-in TLS bypass; all HTTP stacks are isolated mocks."""
from __future__ import annotations
import ssl
import sys
import types
import pytest
import rag.ssl_bypass as module

@pytest.fixture(autouse=True)
def restore_process_state(monkeypatch):
    context = ssl._create_default_https_context
    applied = module._APPLIED
    names = ("CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR")
    env = {name: __import__('os').environ.get(name) for name in names}
    yield
    ssl._create_default_https_context = context
    module._APPLIED = applied
    import os
    for name, value in env.items():
        if value is None: os.environ.pop(name, None)
        else: os.environ[name] = value

@pytest.mark.unit
def test_disabled_setting_is_noop(monkeypatch):
    monkeypatch.setenv("DISABLE_SSL_VERIFY", " false ")
    module._APPLIED = False
    original = ssl._create_default_https_context
    assert module.disable_ssl_verification() is False
    assert ssl._create_default_https_context is original
    assert module._APPLIED is False

@pytest.mark.unit
def test_enabling_patches_http_clients_and_requests_idempotently(monkeypatch, caplog):
    import os
    monkeypatch.setenv("DISABLE_SSL_VERIFY", " TrUe ")
    module._APPLIED = False
    calls = {"http": [], "request": [], "merge": [], "warnings": []}
    class HttpClient:
        def __init__(self, *args, **kwargs): calls["http"].append(kwargs)
    class AsyncClient:
        def __init__(self, *args, **kwargs): calls["http"].append(kwargs)
    httpx = types.ModuleType("httpx"); httpx.Client=HttpClient; httpx.AsyncClient=AsyncClient
    class Session:
        def request(self, method, url, *args, **kwargs): calls["request"].append(kwargs); return "response"
        def merge_environment_settings(self, url, proxies, stream, verify, cert):
            calls["merge"].append(verify); return verify
    requests = types.ModuleType("requests"); requests.sessions=types.SimpleNamespace(Session=Session)
    class Warning: pass
    urllib3=types.ModuleType("urllib3"); urllib3.exceptions=types.SimpleNamespace(InsecureRequestWarning=Warning)
    urllib3.disable_warnings=lambda warning: calls["warnings"].append(warning)
    monkeypatch.setitem(sys.modules, "httpx", httpx)
    monkeypatch.setitem(sys.modules, "requests", requests)
    monkeypatch.setitem(sys.modules, "urllib3", urllib3)
    ssl_original = ssl._create_default_https_context
    assert module.disable_ssl_verification() is True
    assert ssl._create_default_https_context is ssl._create_unverified_context
    assert all(os.environ[name] == "" for name in ("CURL_CA_BUNDLE","REQUESTS_CA_BUNDLE","SSL_CERT_FILE","SSL_CERT_DIR"))
    HttpClient(verify=True, timeout=2); AsyncClient(verify=True)
    assert calls["http"] == [{"verify":False,"timeout":2},{"verify":False}]
    assert Session().request("GET", "https://example.invalid", verify=True) == "response"
    assert calls["request"] == [{"verify":False}]
    assert Session().merge_environment_settings("u", None, False, True, None) is False
    assert calls["merge"] == [False] and calls["warnings"] == [Warning]
    patched_context = ssl._create_default_https_context
    assert module.disable_ssl_verification() is True
    assert ssl._create_default_https_context is patched_context
    assert module._APPLIED is True
    assert "disabled process-wide" in caplog.text

@pytest.mark.unit
def test_import_error_optional_http_dependencies(monkeypatch):
    monkeypatch.setenv("DISABLE_SSL_VERIFY", "1")
    module._APPLIED = False
    original_import = __import__
    def import_without_optional(name, *args, **kwargs):
        if name in {"httpx", "requests", "urllib3"}: raise ImportError(name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", import_without_optional)
    assert module.disable_ssl_verification() is True
