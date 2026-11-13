"""Tests consolidés et isolés pour ``src.ingestion.data_loader``."""
from __future__ import annotations

import hashlib
import json
import runpy
import sys
import types
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion import data_loader as loader


def test_imports_full_src_ingestion_data_loader():
    """Refuse la version tronquée de 4050 octets précédemment testée."""
    source = Path(loader.__file__).resolve()
    assert source.stat().st_size == 8672
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "52216a5e12610da075c0f9a1d58d93c337bde07b9ae842be7485e0a001a942f2"
    )


def test_extract_text_from_txt_reads_utf8(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Basketball", encoding="utf-8")
    assert loader.extract_text_from_txt(path) == "Basketball"


def test_load_text_category_and_empty_text(tmp_path):
    folder = tmp_path / "forum"
    folder.mkdir()
    (folder / "post.txt").write_text("Basketball", encoding="utf-8")
    (folder / "empty.txt").write_text("  \n", encoding="utf-8")
    docs = loader.load_and_parse_files(tmp_path)
    assert docs == [{
        "text": "Basketball", "source": "post.txt", "page": 1, "category": "forum",
        "metadata": {"source": "post.txt", "page": 1, "category": "forum"},
    }]


def test_load_root_text_and_unsupported_file(tmp_path):
    (tmp_path / "root.txt").write_text("  raw text  ", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignored", encoding="utf-8")
    docs = loader.load_and_parse_files(tmp_path)
    if hasattr(loader, "_extract_xlsx"):
        assert docs[0]["text"] == "raw text"
        assert "category" not in docs[0]
        assert "category" not in docs[0]["metadata"]
    else:
        assert docs[0]["text"] == "  raw text  "
        assert docs[0]["category"] is None
        assert docs[0]["metadata"]["category"] is None
    assert len(docs) == 1


def test_load_missing_or_non_directory_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        loader.load_and_parse_files(tmp_path / "missing")
    file_path = tmp_path / "file.txt"
    file_path.write_text("not a directory", encoding="utf-8")
    assert loader.load_and_parse_files(file_path) == []


def test_load_defaults_to_raw_data_dir(monkeypatch, tmp_path):
    (tmp_path / "default.txt").write_text("configured", encoding="utf-8")
    monkeypatch.setattr(loader, "RAW_DATA_DIR", tmp_path)
    assert loader.load_and_parse_files()[0]["text"] == "configured"


def test_load_wraps_text_error(monkeypatch, tmp_path):
    (tmp_path / "broken.txt").write_text("content", encoding="utf-8")
    monkeypatch.setattr(loader, "extract_text_from_txt", lambda path: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(RuntimeError, match="(Erreur de lecture|Error reading) broken.txt: disk full"):
        loader.load_and_parse_files(tmp_path)


def test_ocr_page_returns_empty():
    assert loader._ocr_page(object()) == ""


def test_extract_pdf_ignores_missing_optional_parser(monkeypatch, tmp_path):
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "pymupdf":
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.delitem(sys.modules, "pymupdf", raising=False)
    assert loader._extract_pdf_text(tmp_path / "missing.pdf") == []


def test_extract_pdf_text_extracts_nonempty_pages_and_skips_blank(monkeypatch, tmp_path):
    class Page:
        def __init__(self, text): self.text = text
        def get_text(self, *args): return self.text

    class Pdf:
        def __enter__(self): return [Page(" first "), Page(" ")]
        def __exit__(self, *args): return False

    monkeypatch.setitem(sys.modules, "pymupdf", types.SimpleNamespace(open=lambda path: Pdf()))
    assert loader._extract_pdf_text(tmp_path / "doc.pdf") == [{
        "text": "first", "metadata": {"source": "doc.pdf", "page": 1},
        "source": "doc.pdf", "page": 1,
    }]


def test_extract_pdf_text_uses_ocr_fallback(monkeypatch, tmp_path):
    class Page:
        def get_text(self, *args): return ""

    class Pdf:
        def __enter__(self): return [Page()]
        def __exit__(self, *args): return False

    monkeypatch.setitem(sys.modules, "pymupdf", types.SimpleNamespace(open=lambda path: Pdf()))
    monkeypatch.setattr(loader, "_ocr_page", lambda page: " OCR result ")
    expected = " OCR result " if hasattr(loader, "_extract_xlsx") else "OCR result"
    assert loader._extract_pdf_text(tmp_path / "scan.pdf")[0]["text"] == expected


def test_extract_pdf_propagates_parser_errors(tmp_path):
    class BrokenPdf:
        def __enter__(self): raise OSError("parser failed")
        def __exit__(self, *args): return False

    sys.modules["pymupdf"] = types.SimpleNamespace(open=lambda path: BrokenPdf())
    try:
        if hasattr(loader, "_extract_xlsx"):
            with pytest.raises(RuntimeError, match="Unable to extract PDF.*parser failed"):
                loader._extract_pdf_text(tmp_path / "broken.pdf")
        else:
            with pytest.raises(OSError, match="parser failed"):
                loader._extract_pdf_text(tmp_path / "broken.pdf")
    finally:
        sys.modules.pop("pymupdf", None)


def test_loader_skips_txt_when_all_pdfs_are_empty(monkeypatch, tmp_path):
    (tmp_path / "note.txt").write_text("corpus", encoding="utf-8")
    (tmp_path / "scan.pdf").touch()
    monkeypatch.setattr(loader, "_extract_pdf_text", lambda path: [])
    docs = loader.load_and_parse_files(tmp_path)
    if hasattr(loader, "_extract_xlsx"):
        assert len(docs) == 1 and docs[0]["text"] == "corpus"
    else:
        assert docs == []


def test_loader_includes_pdf_and_txt_when_pdf_has_content(monkeypatch, tmp_path):
    (tmp_path / "note.txt").write_text("corpus", encoding="utf-8")
    (tmp_path / "scan.pdf").touch()
    monkeypatch.setattr(loader, "_extract_pdf_text", lambda path: [
        {"text": "pdf text", "metadata": {"source": "scan.pdf", "page": 1},
         "source": "scan.pdf", "page": 1}
    ])
    docs = loader.load_and_parse_files(tmp_path)
    assert len(docs) == 2
    assert docs[0]["text"] == "corpus"
    if hasattr(loader, "_extract_xlsx"):
        assert "category" not in docs[1]
        assert "category" not in docs[1]["metadata"]
    else:
        assert docs[1]["category"] is None
        assert docs[1]["metadata"]["category"] is None


def test_loader_adds_pdf_category_and_handles_missing_metadata(monkeypatch, tmp_path):
    folder = tmp_path / "articles"
    folder.mkdir()
    (folder / "paper.pdf").touch()
    monkeypatch.setattr(loader, "_extract_pdf_text", lambda path: [{"text": "pdf"}])
    docs = loader.load_and_parse_files(tmp_path)
    if hasattr(loader, "_extract_xlsx"):
        assert docs == [{"text": "pdf", "metadata": {"category": "articles"}, "category": "articles"}]
    else:
        assert docs == [{"text": "pdf", "category": "articles"}]


def test_loader_wraps_pdf_exception_from_file_loop(monkeypatch, tmp_path):
    (tmp_path / "paper.pdf").touch()
    calls = 0

    def fail_on_second_call(path):
        nonlocal calls
        calls += 1
        if calls == 1 and not hasattr(loader, "_extract_xlsx"):
            return []
        raise OSError("read error")

    monkeypatch.setattr(loader, "_extract_pdf_text", fail_on_second_call)
    with pytest.raises(RuntimeError, match="(Erreur de lecture|Error reading) paper.pdf: read error"):
        loader.load_and_parse_files(tmp_path)


def test_extract_xlsx_rows_formats_values_and_skips_empty_rows(monkeypatch, tmp_path):
    frame = pd.DataFrame([{"player": "A", "points": 2}, {"player": None, "points": None}])
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: {"Stats": frame})
    rows = loader._extract_xlsx_rows(tmp_path / "stats.xlsx")
    assert len(rows) == 1
    assert rows[0]["text"] == "player: A | points: 2.0"
    assert rows[0]["source"] == "stats.xlsx#Stats" and rows[0]["page"] == 2
    assert rows[0]["metadata"]["source"] == "stats.xlsx#Stats"
    if hasattr(loader, "_extract_xlsx"):
        assert rows[0]["metadata"]["sheet"] == "Stats"


def test_extract_xlsx_rows_multiple_sheets_and_empty_mapping(monkeypatch, tmp_path):
    data = {"One": pd.DataFrame([{"a": 1}, {"a": 2}]), "Two": pd.DataFrame([{"b": "x"}])}
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: data)
    rows = loader._extract_xlsx_rows(tmp_path / "book.xlsx")
    assert [(d["source"], d["page"]) for d in rows] == [
        ("book.xlsx#One", 2), ("book.xlsx#One", 3), ("book.xlsx#Two", 2)
    ]
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: {})
    assert loader._extract_xlsx_rows(tmp_path / "empty.xlsx") == []


def test_loader_reads_xlsx_and_wraps_read_excel_error(monkeypatch, tmp_path):
    (tmp_path / "table.xlsx").touch()
    frame = pd.DataFrame([{"team": "A"}])
    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: {"Teams": frame})
    docs = loader.load_and_parse_files(tmp_path)
    assert docs[0]["source"] == "table.xlsx#Teams"
    if hasattr(loader, "_extract_xlsx"):
        assert "category" not in docs[0]
        assert docs[0]["metadata"]["sheet"] == "Teams"
    else:
        assert "category" not in docs[0]

    monkeypatch.setattr(pd, "read_excel", lambda *a, **k: (_ for _ in ()).throw(OSError("bad workbook")))
    with pytest.raises(RuntimeError, match="(Erreur de lecture|Error reading) table.xlsx:.*bad workbook"):
        loader.load_and_parse_files(tmp_path)


def test_zip_extracts_safe_content_and_creates_destination(tmp_path):
    archive_path = tmp_path / "good.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/file.txt", "safe")
    destination = tmp_path / "out"
    assert loader.download_and_extract_zip(archive_path, destination) == destination
    assert (destination / "nested/file.txt").read_text(encoding="utf-8") == "safe"


def test_zip_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escaped.txt", "unsafe")
    with pytest.raises(ValueError, match="(Chemin dangereux|Unsafe path)"):
        loader.download_and_extract_zip(archive_path, tmp_path / "out")


def test_processed_json_save_load_defaults_and_roundtrip(monkeypatch, tmp_path):
    docs = [{"text": "ok"}]
    custom = loader.save_processed_documents(docs, tmp_path / "nested/docs.json")
    assert loader.load_processed_documents(custom) == docs
    processed = tmp_path / "processed"
    if hasattr(loader, "PROCESSED_DATA_DIR"):
        monkeypatch.setattr(loader, "PROCESSED_DATA_DIR", processed)
    else:
        monkeypatch.setattr(loader.config, "PROCESSED_DATA_DIR", processed)
    default = loader.save_processed_documents(docs)
    assert default == tmp_path / "processed" / "documents.json"
    assert loader.load_processed_documents() == docs


def test_processed_json_missing_and_malformed(tmp_path):
    with pytest.raises(FileNotFoundError):
        loader.load_processed_documents(tmp_path / "missing.json")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        loader.load_processed_documents(malformed)


@pytest.mark.skipif(not hasattr(loader, "_LoaderModule"), reason="compatibility shim absent")
def test_loader_module_assignment_compatibility():
    module = types.ModuleType("temporary_loader")
    module.__class__ = loader._LoaderModule
    module.load_and_parse_files = lambda: "canonical"
    legacy = lambda: "legacy"
    legacy.__name__ = "_legacy_parse"
    module.load_and_parse_files = legacy
    assert module.load_and_parse_files() == "canonical"
    replacement = lambda: "replacement"
    module.load_and_parse_files = replacement
    assert module.load_and_parse_files() == "replacement"


def test_module_main_loads_and_saves_documents(monkeypatch, tmp_path, capsys):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    (raw / "main.txt").write_text("main document", encoding="utf-8")
    # Configurations in the project have used both RAW_DATA_DIR / PROCESSED_DATA_DIR
    # and DATA_RAW_DIR / DATA_PROCESSED_DIR. Patch the names that actually exist,
    # so runpy's fresh execution of data_loader reads the temporary test folders.
    raw_config_name = next(
        name for name in ("RAW_DATA_DIR", "DATA_RAW_DIR", "INPUT_DIR")
        if hasattr(loader.config, name)
    )
    processed_config_name = next(
        name for name in ("PROCESSED_DATA_DIR", "DATA_PROCESSED_DIR", "PROCESSED_DIR")
        if hasattr(loader.config, name)
    )
    monkeypatch.setattr(loader.config, raw_config_name, raw)
    monkeypatch.setattr(loader.config, processed_config_name, processed)
    runpy.run_path(loader.__file__, run_name="__main__")
    output = processed / "documents.json"
    assert json.loads(output.read_text(encoding="utf-8"))[0]["text"] == "main document"
    assert "1 documents saved to" in capsys.readouterr().out
