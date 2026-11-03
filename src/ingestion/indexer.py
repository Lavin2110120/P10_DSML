"""Build the FAISS index from the processed document corpus."""
from __future__ import annotations

import argparse
import logging

from rag.ssl_bypass import disable_ssl_verification
disable_ssl_verification()

from ingestion.data_loader import load_processed_documents
from llm.mistral_client import MistralClient
from rag import config
from rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

_PLACEHOLDER_API_KEYS = {
    "test-key",
    "changeme",
    "your_api_key",
    "your-api-key",
    "your-mistral-api-key",
    "replace-me",
    "replace_me",
    "none",
    "null",
}


def _configured_api_key() -> str:
    """Return the configured Mistral key or fail with a useful setup message."""
    value = getattr(config, "MISTRAL_API_KEY", None)
    api_key = str(value).strip() if value is not None else ""
    if not api_key or api_key.lower() in _PLACEHOLDER_API_KEYS:
        raise RuntimeError(
            "MISTRAL_API_KEY is missing or still set to a placeholder. "
            "Set a valid MISTRAL_API_KEY in the environment or .env file."
        )
    return api_key


def index_documents(input_path: str | None = None) -> VectorStoreManager:
    """Load processed documents, embed them with Mistral, and build the index."""
    documents = load_processed_documents(input_path)
    client = MistralClient(api_key=_configured_api_key())
    manager = VectorStoreManager(client=client)
    manager.build_index(documents)

    indexed_count = getattr(getattr(manager, "index", None), "ntotal", 0) or 0
    logger.info("Indexed %s chunks", indexed_count)
    return manager


def main() -> VectorStoreManager:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="Path to processed documents JSON")
    args = parser.parse_args()
    return index_documents(args.input)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
