"""Comportement observable du bypass TLS, sans connexion réseau."""
from __future__ import annotations

import importlib
import os
import ssl
import sys
import types
import warnings

import dotenv
import pytest

_ENV_NAMES = (
    "DISABLE_SSL_VERIFY",
    "PYTHONHTTPSVERIFY",
    "CURL_CA_BUNDLE",
    "REQUESTS_CA_BUNDLE",
    "GRPC_DEFAULT_SSL_ROOTS_FILE_PATH",
)


def _noop_load_dotenv(*args, **kwargs):
    """Empêche le .env réel d'écraser la configuration isolée de chaque test."""
    return False


# Empêche l'import initial d'appliquer le bypass de manière globale
_previous_flag = os.environ.get("DISABLE_SSL_VERIFY")
_previous_load_dotenv = dotenv.load_dotenv
os.environ["DISABLE_SSL_VERIFY"] = "0"
dotenv.load_dotenv = _noop_load_dotenv
try:
    import src.ssl_bypass as module
finally:
    dotenv.load_dotenv = _previous_load_dotenv
    if _previous_flag is None:
        os.environ.pop("DISABLE_SSL_VERIFY", None)
    else:
        os.environ["DISABLE_SSL_VERIFY"] = _previous_flag


@pytest.fixture(autouse=True)
def isolate_process_state(monkeypatch):
    """Isole et réinitialise l'état TLS et les variables d'environnement."""
    sentinel_context = lambda: None
    monkeypatch.setattr(ssl, "_create_default_https_context", sentinel_context)

    for name in _ENV_NAMES:
        if name == "DISABLE_SSL_VERIFY":
            monkeypatch.setenv(name, "0")
        else:
            monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(dotenv, "load_dotenv", _noop_load_dotenv)
    yield


@pytest.mark.parametrize("value", ["0", "false", "NO", " off ", ""])
def test_disabled_mode_leaves_tls_and_environment_untouched(monkeypatch, value):
    monkeypatch.setenv("DISABLE_SSL_VERIFY", value)
    context_before = ssl._create_default_https_context
    environment_before = {name: os.environ.get(name) for name in _ENV_NAMES}

    # Le mode désactivé est régi lors du chargement/import du module
    importlib.reload(module)

    assert ssl._create_default_https_context is context_before
    assert {name: os.environ.get(name) for name in _ENV_NAMES} == environment_before


def test_enabled_mode_patches_tls_clients_environment_and_is_idempotent(monkeypatch):
    monkeypatch.setenv("DISABLE_SSL_VERIFY", "1")
    original_ssl_unverified = ssl._create_unverified_context

    # Capture l'appel à disable_warnings de urllib3
    warnings_called = []
    urllib3_mock = types.ModuleType("urllib3")
    warning_cls = type("InsecureRequestWarning", (Warning,), {})
    urllib3_mock.exceptions = types.SimpleNamespace(InsecureRequestWarning=warning_cls)
    urllib3_mock.disable_warnings = lambda w: warnings_called.append(w)
    monkeypatch.setitem(sys.modules, "urllib3", urllib3_mock)

    # L'activation par env se produit au chargement ou via l'appel de fonction
    importlib.reload(module)

    # 1. Contexte SSL standard non vérifié
    assert ssl._create_default_https_context is original_ssl_unverified

    # 2. Variables d'environnement définies par src/ssl_bypass.py
    assert os.environ.get("PYTHONHTTPSVERIFY") == "0"
    assert os.environ.get("GRPC_DEFAULT_SSL_ROOTS_FILE_PATH") == ""
    assert os.environ.get("CURL_CA_BUNDLE") == ""
    assert os.environ.get("REQUESTS_CA_BUNDLE") == ""

    # 3. Idempotence : un second appel explicite ne casse rien
    module.disable_ssl_verification()
    assert ssl._create_default_https_context is original_ssl_unverified
    assert os.environ.get("PYTHONHTTPSVERIFY") == "0"


@pytest.mark.parametrize("value", ["0", "false", "no", "NO", " off ", ""])
def test_import_with_falsy_env_does_not_apply_bypass(monkeypatch, value):
    monkeypatch.setenv("DISABLE_SSL_VERIFY", value)
    context_before = ssl._create_default_https_context

    importlib.reload(module)

    assert ssl._create_default_https_context is context_before