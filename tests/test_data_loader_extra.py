from __future__ import annotations
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace
import pytest
import rag_test_support  # noqa: F401 - injecte les SDK optionnels avant import des sources
import src.ingestion.data_loader as loader
_native_parse = loader.load_and_parse_files
# Snapshot avant adaptateur de compatibilité chargé par les anciens tests.

@pytest.mark.unit
def test_zip_extraction_and_zip_slip_rejection(tmp_path):
    archive_path = tmp_path / "valid.zip"
    with zipfile.ZipFile(archive_path, "w") as archive: archive.writestr("nested/doc.txt", "ok")
    destination = tmp_path / "out"
    assert loader.download_and_extract_zip(archive_path, destination) == destination
    assert (destination / "nested" / "doc.txt").read_text() == "ok"
    malicious = tmp_path / "bad.zip"
    with zipfile.ZipFile(malicious, "w") as archive: archive.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="dangereux"): loader.download_and_extract_zip(malicious, destination)

@pytest.mark.unit
def test_pdf_pages_extract_text_and_ocr_fallback(monkeypatch, tmp_path):
    path = tmp_path / "pages.pdf"; path.touch()
    class Page:
        def __init__(self, text): self.value = text
        def get_text(self): return self.value
    class Doc:
        def __enter__(self): return iter([Page(" visible "), Page("")])
        def __exit__(self, *args): return False
    monkeypatch.setitem(__import__('sys').modules, "pymupdf", SimpleNamespace(open=lambda path: Doc()))
    monkeypatch.setattr(loader, "_ocr_page", lambda page: "scanned text")
    docs = loader._extract_pdf_text(path)
    assert [d["text"] for d in docs] == ["visible", "scanned text"]
    assert docs[1]["metadata"] == {"source":"pages.pdf", "page":2}

@pytest.mark.unit
def test_xlsx_rows_and_empty_values(monkeypatch, tmp_path):
    import pandas as pd
    path = tmp_path / "stats.xlsx"; path.touch()
    monkeypatch.setattr(pd, "read_excel", lambda *a, **kw: {"Players": pd.DataFrame([["LeBron", 25], [None, None], ["Davis", None]], columns=["name", "points"])})
    docs = loader._extract_xlsx_rows(path)
    assert [d["text"] for d in docs] == ["name: LeBron | points: 25.0", "name: Davis"]
    assert docs[0]["source"] == "stats.xlsx#Players" and docs[0]["page"] == 2

@pytest.mark.unit
def test_loader_directory_filter_failures_and_missing(tmp_path, monkeypatch):
    nested = tmp_path / "folder"; nested.mkdir()
    (tmp_path / "ignore.txt").write_text("skip", encoding="utf8")
    (tmp_path / "empty.pdf").touch()
    monkeypatch.setattr(loader, "_extract_pdf_text", lambda path: [] if path.name == "empty.pdf" else (_ for _ in ()).throw(UnicodeError("bad encoding")))
    assert loader.load_and_parse_files(tmp_path) in ([], [{"page_content": "skip", "metadata": {"filename": "ignore.txt", "source": "ignore.txt"}}])
    (tmp_path / "ignore.txt").unlink(missing_ok=True)
    (tmp_path / "corrupt.xlsx").touch()
    monkeypatch.setattr(loader, "_extract_xlsx_rows", lambda path: (_ for _ in ()).throw(ValueError("corrupt")))
    with pytest.raises(RuntimeError, match="corrupt.xlsx"): loader.load_and_parse_files(tmp_path)
    with pytest.raises(FileNotFoundError): loader.load_and_parse_files(tmp_path / "missing")
    assert loader.load_and_parse_files(tmp_path / "folder") == []

@pytest.mark.unit
def test_processed_json_round_trip_and_errors(tmp_path):
    output = tmp_path / "processed" / "docs.json"
    expected = [{"text": "équipe NBA", "metadata": {"source":"x"}}]
    assert loader.save_processed_documents(expected, output) == output
    assert loader.load_processed_documents(output) == expected
    assert "équipe" in output.read_text(encoding="utf-8")
    with pytest.raises(FileNotFoundError): loader.load_processed_documents(tmp_path / "no.json")
    output.write_text("not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError): loader.load_processed_documents(output)

@pytest.mark.unit
def test_pdf_missing_parser_and_nested_pdf_category(monkeypatch,tmp_path):
    import sys
    path=tmp_path/"x.pdf"; path.touch()
    monkeypatch.setitem(sys.modules,"pymupdf",None)
    assert loader._extract_pdf_text(path)==[]
    nested=tmp_path/"nba"; nested.mkdir(); (nested/"ok.pdf").touch()
    monkeypatch.setitem(sys.modules,"pymupdf",SimpleNamespace(open=lambda *_: iter([])))
    monkeypatch.setattr(loader,"_extract_pdf_text",lambda p:[{"text":"t","metadata":{"source":"ok.pdf"}}])
    docs=loader.load_and_parse_files(tmp_path)
    assert docs[0]["category"]=="nba" and docs[0]["metadata"]["category"]=="nba"
