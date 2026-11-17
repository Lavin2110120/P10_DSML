"""Load raw text, PDF, and Excel files into a normalized document corpus.

Each extracted document has a ``text`` field plus ``source``, ``page``, and
``metadata`` fields so ingestion and indexing consumers share one schema.
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

try:
    from ..models import Document, Chunk
    from ..observability import span
except ImportError:
    from src.models import Document, Chunk
    from src.observability import span

# This module lives in src/ingestion; apply the project's explicit TLS setting
# before optional parsers or other consumers initialize network clients.
try:
    from .. import config
    from ..ssl_bypass import disable_ssl_verification
except ImportError:  # Support imports when the repository package is src.*
    from src import config
    from src.ssl_bypass import disable_ssl_verification

if os.getenv("DISABLE_SSL_VERIFY", "0") == "1":
    disable_ssl_verification()

_LOG = logging.getLogger(__name__)

# Match the current config.py names and retain compatibility with config
# variants that expose DATA_RAW_DIR / DATA_PROCESSED_DIR or INPUT_DIR.
RAW_DATA_DIR = Path(
    next(
        getattr(config, name)
        for name in ("RAW_DATA_DIR", "DATA_RAW_DIR", "INPUT_DIR")
        if hasattr(config, name)
    )
)
PROCESSED_DATA_DIR = Path(
    next(
        getattr(config, name)
        for name in ("PROCESSED_DATA_DIR", "DATA_PROCESSED_DIR", "PROCESSED_DIR")
        if hasattr(config, name)
    )
)
# Public aliases used by a few older callers.
INPUT_DIR = RAW_DATA_DIR
PROCESSED_DIR = PROCESSED_DATA_DIR


def _chunk_settings(
    chunk_size: int | None = None, overlap: int | None = None
) -> tuple[int, int]:
    """Return validated chunk settings, defaulting to config.py values."""
    size = getattr(config, "CHUNK_SIZE") if chunk_size is None else chunk_size
    overlap_size = getattr(config, "CHUNK_OVERLAP") if overlap is None else overlap
    if not isinstance(size, int) or size <= 0:
        raise ValueError("CHUNK_SIZE doit être un entier positif")
    if not isinstance(overlap_size, int) or overlap_size < 0:
        raise ValueError("CHUNK_OVERLAP doit être un entier positif ou nul")
    if overlap_size >= size:
        raise ValueError("CHUNK_OVERLAP doit être inférieur à CHUNK_SIZE")
    return size, overlap_size


def normalize_text(text: str) -> str:
    """Nettoyer les caractères parasites et normaliser les espaces du texte."""
    if not isinstance(text, str):
        text = str(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    # Supprimer les caractères de contrôle, de format (dont zero-width/BOM),
    # non imprimables et autres caractères Unicode non textuels.
    text = "".join(
        char for char in text
        if char == "\n" or (char.isprintable() and unicodedata.category(char) not in {"Cc", "Cf", "Cs", "Cn"})
    )
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    normalized_lines: list[str] = []
    blank_seen = False
    for line in lines:
        if line:
            normalized_lines.append(line)
            blank_seen = False
        elif normalized_lines and not blank_seen:
            normalized_lines.append("")
            blank_seen = True
    return "\n".join(normalized_lines).strip()


def _preferred_chunk_end(text: str, start: int, hard_end: int) -> int:
    """Prefer sentence/word boundaries close to the maximum permitted end."""
    if hard_end >= len(text):
        return len(text)
    floor = start + max(1, int((hard_end - start) * 0.8))
    sentence_ends = [
        match.end() for match in re.finditer(r"[.!?](?=\s|$)", text[start:hard_end])
        if start + match.end() >= floor
    ]
    if sentence_ends:
        return start + sentence_ends[-1]
    spaces = [match.start() + start for match in re.finditer(r"\s+", text[start:hard_end])
              if start + match.start() >= floor]
    if spaces:
        return spaces[-1]
    return hard_end


def chunk_text(
    text: str, chunk_size: int | None = None, overlap: int | None = None
) -> list[str]:
    """Découper du texte normalisé en segments bornés, avec chevauchement."""
    size, overlap_size = _chunk_settings(chunk_size, overlap)
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        hard_end = min(start + size, len(cleaned))
        end = _preferred_chunk_end(cleaned, start, hard_end)
        # Keep boundary whitespace inside the slice: it preserves exact source
        # coverage and makes the configured character overlap deterministic.
        chunk = cleaned[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = max(start + 1, end - overlap_size)
    return chunks


def chunk_document(
    document: dict[str, Any], chunk_size: int | None = None, overlap: int | None = None
) -> list[dict[str, Any]]:
    """Chunk one source record while retaining its fields and source metadata.

    ``text`` is the canonical loader schema; accept common legacy document
    fields as fallbacks so indexing does not silently discard valid content.
    """
    source_text = document.get("text")
    if source_text is None:
        source_text = document.get("page_content")
    if source_text is None:
        source_text = document.get("content")
    if source_text is None:
        _LOG.warning("Document ignoré : aucun champ texte reconnu (text/page_content/content)")
        return []
    chunks = chunk_text(source_text, chunk_size, overlap)
    output: list[dict[str, Any]] = []
    for index, text in enumerate(chunks):
        item = dict(document)
        item["text"] = normalize_text(text)
        metadata = dict(document.get("metadata") or {})
        for key in ("source", "page"):
            if key in document and key not in metadata:
                metadata[key] = document[key]
        metadata["chunk_index"] = index
        item["metadata"] = metadata
        item["chunk_index"] = index
        try:
            validated = Chunk.model_validate(item)
            output.append(validated.model_dump(mode="python"))
        except Exception as exc:
            _LOG.warning("Chunk %d ignoré : validation impossible (%s)", index, exc)
    return output


def chunk_documents(
    documents: list[dict[str, Any]], chunk_size: int | None = None, overlap: int | None = None
) -> list[dict[str, Any]]:
    """Normalize and chunk all document records for embedding/indexing."""
    size, overlap_size = _chunk_settings(chunk_size, overlap)
    output: list[dict[str, Any]] = []
    normalized_count = 0
    for index, document in enumerate(documents):
        if not isinstance(document, dict) or not any(document.get(key) is not None for key in ("text", "page_content", "content")):
            _LOG.warning("Document ignoré : aucun champ texte reconnu (text/page_content/content)")
            continue
        try:
            validated = Document.model_validate(document)
            validated.text = normalize_text(validated.text)
            validated = Document.model_validate(validated.model_dump(mode="python", exclude_none=True))
            normalized_count += 1
            output.extend(chunk_document(validated.model_dump(mode="python", exclude_none=True), size, overlap_size))
        except Exception as exc:
            _LOG.warning("Document %d ignoré avant découpage : validation impossible (%s)", index, exc)
    _LOG.info("Normalisation et découpage : %d documents -> %d chunks", normalized_count, len(output))
    return output


def extract_text_from_txt(file_path: str | Path) -> str:
    """Read a UTF-8 text file, accepting an optional UTF-8 BOM."""
    return Path(file_path).read_text(encoding="utf-8-sig")


def _ocr_page(page: Any) -> str:
    """OCR a PyMuPDF page when pytesseract is installed; otherwise return ''."""
    try:
        import pytesseract
        from PIL import Image

        pixmap = page.get_pixmap()
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        return pytesseract.image_to_string(image).strip()
    except Exception as exc:
        _LOG.debug("OCR unavailable for PDF page: %s", exc)
        return ""


def _extract_pdf_text(path: str | Path) -> list[dict[str, Any]]:
    """Extract each non-empty PDF page as an individual document record."""
    source_path = Path(path)
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz  # type: ignore[no-redef]
    except ImportError:
        _LOG.warning("PDF extraction skipped; PyMuPDF is not installed (%s)", source_path)
        return []

    documents: list[dict[str, Any]] = []
    try:
        with fitz.open(str(source_path)) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = (page.get_text("text") or "").strip()
                if not text:
                    text = _ocr_page(page)
                if not text:
                    continue
                metadata = {"source": source_path.name, "page": page_number}
                documents.append(
                    {
                        "text": text,
                        "source": source_path.name,
                        "page": page_number,
                        "metadata": metadata,
                    }
                )
    except Exception as exc:
        raise RuntimeError(f"Unable to extract PDF {source_path}: {exc}") from exc
    return documents


def _extract_xlsx(path: str | Path) -> list[dict[str, Any]]:
    """Convert non-empty worksheet rows to normalized document records."""
    import pandas as pd

    workbook_path = Path(path)
    try:
        sheets = pd.read_excel(workbook_path, sheet_name=None)
    except Exception as exc:
        raise RuntimeError(f"Unable to read Excel workbook {workbook_path}: {exc}") from exc

    documents: list[dict[str, Any]] = []
    for sheet_name, frame in sheets.items():
        for row_number, (_, row) in enumerate(frame.iterrows(), start=2):
            fields: list[str] = []
            for column, value in row.items():
                if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
                    continue
                fields.append(f"{column}: {value}")
            if not fields:
                continue

            source = f"{workbook_path.name}#{sheet_name}"
            metadata = {"source": source, "page": row_number, "sheet": str(sheet_name)}
            documents.append(
                {
                    "text": " | ".join(fields),
                    "source": source,
                    "page": row_number,
                    "metadata": metadata,
                }
            )
    return documents


def _extract_xlsx_rows(path: str | Path) -> list[dict[str, Any]]:
    """Backward-compatible alias for :func:`_extract_xlsx`."""
    return _extract_xlsx(path)


def download_and_extract_zip(
    archive_path: str | Path, destination: str | Path
) -> Path:
    """Extract a ZIP archive while rejecting paths that escape destination."""
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    root = destination_path.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination_path / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        archive.extractall(destination_path)
    return destination_path


def _load_and_parse_files(input_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Recursively load supported files from ``input_dir`` or configured raw data."""
    root = Path(input_dir) if input_dir is not None else RAW_DATA_DIR
    if not root.exists():
        raise FileNotFoundError(str(root))
    if not root.is_dir():
        return []

    documents: list[dict[str, Any]] = []
    supported = {".txt", ".pdf", ".xlsx"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        relative_path = path.relative_to(root)
        category = relative_path.parts[0] if len(relative_path.parts) > 1 else None
        try:
            if path.suffix.lower() == ".txt":
                text = extract_text_from_txt(path).strip()
                extracted = []
                if text:
                    metadata = {"source": path.name, "page": 1}
                    if category is not None:
                        metadata["category"] = category
                    extracted.append(
                        {
                            "text": text,
                            "source": path.name,
                            "page": 1,
                            "metadata": metadata,
                        }
                    )
            elif path.suffix.lower() == ".pdf":
                extracted = _extract_pdf_text(path)
            else:
                extracted = _extract_xlsx(path)

            for document in extracted:
                metadata = document.setdefault("metadata", {})
                if category is not None:
                    document["category"] = category
                    metadata["category"] = category
                try:
                    Document.model_validate(document)
                except Exception as validation_error:
                    _LOG.warning("Document ignoré après validation (%s) : %s", path.name, validation_error)
                    continue
                documents.append(document)
        except Exception as exc:
            raise RuntimeError(f"Error reading {path.name}: {exc}") from exc
    return documents


def load_and_parse_files(input_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Charge les fichiers sous une trace Logfire facultative."""
    with span("rag.load_documents"):
        return _load_and_parse_files(input_dir)


def save_processed_documents(
    documents: list[dict[str, Any]], output_path: str | Path | None = None
) -> Path:
    """Serialize extracted documents to the configured processed-data folder."""
    path = Path(output_path) if output_path is not None else PROCESSED_DATA_DIR / "documents.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def load_processed_documents(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load the JSON document corpus consumed by the indexer."""
    source_path = Path(path) if path is not None else PROCESSED_DATA_DIR / "documents.json"
    if not source_path.exists():
        raise FileNotFoundError(str(source_path))
    with source_path.open("r", encoding="utf-8") as source_file:
        documents = json.load(source_file)
    if not isinstance(documents, list):
        raise ValueError(f"Processed document file must contain a JSON list: {source_path}")
    valid = []
    for index, item in enumerate(documents):
        try:
            Document.model_validate(item)
            valid.append(item)
        except Exception as exc:
            _LOG.warning("Document %d ignoré dans %s : validation impossible (%s)", index, source_path.name, exc)
    return valid


if __name__ == "__main__":
    loaded_documents = load_and_parse_files()
    output_file = save_processed_documents(loaded_documents)
    print(f"{len(loaded_documents)} documents saved to {output_file}")
