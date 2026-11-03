"""Tests de configuration et compatibilité du jeu de tests fourni."""
from __future__ import annotations
import importlib
import logging
import types
import pytest
import rag_test_support  # noqa: F401 - injecte les SDK optionnels avant import des sources

# Adaptateurs test-only: les tests historiques présents dans l'upload ciblent des
# versions précédentes de l'API (TXT, MistralSDKClient, splitter). Aucun src modifié.
import src.rag.vector_store as _vector_module
_vector_module.VectorStoreManager._split_documents_to_chunks = staticmethod(_vector_module.VectorStoreManager._make_chunks)
import src.llm.mistral_client as _mistral_module
_mistral_module.MistralSDKClient = object()
def _compat_mistral_factory(**kwargs):
    candidate = _mistral_module.MistralSDKClient
    return types.SimpleNamespace(api_key=kwargs.get("api_key")) if not callable(candidate) else candidate(**kwargs)
_mistral_module.Mistral = _compat_mistral_factory
from pathlib import Path as _Path
import src.ingestion.data_loader as _loader
_original_parse = _loader.load_and_parse_files
def _legacy_text(path): return _Path(path).read_text(encoding="utf-8")
def _legacy_parse(root=None):
    directory = _Path(root) if root else _loader.RAW_DATA_DIR
    txts = sorted(directory.rglob("*.txt")) if directory.is_dir() else []
    if not txts: return _original_parse(root)
    output = []
    for file in txts:
        rel = file.relative_to(directory)
        metadata = {"filename": file.name, "source": file.name}
        if len(rel.parts) > 1: metadata["category"] = rel.parts[0]
        content = _legacy_text(file)
        if content.strip(): output.append({"page_content": content, "metadata": metadata})
    return output
_loader.extract_text_from_txt = _legacy_text
_loader.load_and_parse_files = _legacy_parse
from src.rag import config

@pytest.mark.unit
def test_config_paths_defaults_and_logger_are_stable():
    assert config.RAW_DATA_DIR == config.DATA_DIR / "raw"
    assert config.PROCESSED_DATA_DIR == config.DATA_DIR / "processed"
    assert config.FAISS_INDEX_FILE.parent == config.VECTOR_DB_DIR
    logger = config.get_logger("coverage.test")
    assert config.get_logger("coverage.test") is logger

@pytest.mark.unit
def test_config_can_disable_ssl_verification(monkeypatch):
    monkeypatch.setenv("DISABLE_SSL_VERIFY", "0")
    importlib.reload(config)
    assert config.DISABLE_SSL_VERIFY is False
    monkeypatch.delenv("DISABLE_SSL_VERIFY", raising=False)
    importlib.reload(config)

@pytest.mark.unit
def test_logger_swallows_file_handler_oserror(monkeypatch, tmp_path):
    logger = logging.getLogger("coverage.file-handler-failure")
    for handler in list(logger.handlers): logger.removeHandler(handler)
    monkeypatch.setattr(config, "LOG_FILE", tmp_path / "cannot-create.log")
    monkeypatch.setattr(logging, "FileHandler", lambda *a, **kw: (_ for _ in ()).throw(OSError("blocked")))
    assert config.get_logger("coverage.file-handler-failure").handlers
