"""Tests des fonctions de chargement de documents."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.data_loader import extract_text_from_txt, load_and_parse_files


@pytest.mark.unit
class TestDataLoader:
    def test_extract_text_from_txt(self, tmp_path: Path) -> None:
        content = "Contenu de test\nDeuxième ligne."
        source = tmp_path / "document.txt"
        source.write_text(content, encoding="utf-8")

        assert extract_text_from_txt(str(source)) == content

    def test_load_and_parse_files_returns_document_metadata(self, tmp_path: Path) -> None:
        content = "Un document de test."
        (tmp_path / "document.txt").write_text(content, encoding="utf-8")

        documents = load_and_parse_files(str(tmp_path))

        assert len(documents) == 1
        assert documents[0]["page_content"] == content
        assert documents[0]["metadata"]["filename"] == "document.txt"
        assert documents[0]["metadata"]["source"] == "document.txt"

    def test_unsupported_file_is_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "notes.md").write_text("non pris en charge", encoding="utf-8")

        assert load_and_parse_files(str(tmp_path)) == []


@pytest.mark.integration
def test_loader_reads_a_directory_tree(tmp_path: Path) -> None:
    nested = tmp_path / "categorie"
    nested.mkdir()
    (nested / "document.txt").write_text("Texte imbriqué", encoding="utf-8")

    documents = load_and_parse_files(str(tmp_path))

    assert len(documents) == 1
    assert documents[0]["metadata"]["category"] == "categorie"
