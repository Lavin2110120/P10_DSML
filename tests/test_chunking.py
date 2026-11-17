from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.ingestion.data_loader import normalize_text, chunk_text, chunk_documents, _chunk_settings


def test_normalization_and_short_text():
    sample = "\ufeff A\t\tB\r\n\r\n\r\nC\u200b\x00\x07  D \n"
    assert normalize_text(sample) == "A B\n\nC D"
    short = "court texte sans perte"
    assert chunk_text(short) == [short]


def test_config_bounds_and_exact_overlap_without_boundaries():
    assert CHUNK_OVERLAP < CHUNK_SIZE
    try:
        _chunk_settings(CHUNK_SIZE, CHUNK_SIZE)
    except ValueError:
        pass
    else:
        raise AssertionError("overlap >= size should be rejected")

    source = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz"
    parts = chunk_text(source, chunk_size=20, overlap=5)
    assert all(len(part) <= 20 for part in parts)
    assert "".join([parts[0]] + [part[5:] for part in parts[1:]]) == source
    assert all(left[-5:] == right[:5] for left, right in zip(parts, parts[1:]))


def test_sentence_chunks_bounded_and_metadata_preserved():
    text = " ".join(f"phrase{i} contient quelques mots." for i in range(100))
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(part) <= CHUNK_SIZE for part in chunks)
    cursor = 0
    for previous, following in zip(chunks, chunks[1:]):
        pos = text.find(previous, cursor)
        assert pos >= 0
        end = pos + len(previous)
        next_pos = text.find(following, end - CHUNK_OVERLAP)
        assert next_pos == end - CHUNK_OVERLAP
        assert previous[-CHUNK_OVERLAP:] == following[:CHUNK_OVERLAP]
        cursor = pos + 1

    records = chunk_documents([{"text": text, "source": "a.pdf", "page": 2, "metadata": {"kind": "pdf"}}])
    assert all(record["source"] == "a.pdf" and record["page"] == 2 for record in records)
    assert all(record["metadata"]["source"] == "a.pdf" for record in records)
    assert [r["chunk_index"] for r in records] == list(range(len(records)))
