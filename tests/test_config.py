import src.config as config


def test_gemini_config_paths_dimension_and_defaults():
    assert config.EMBEDDING_DIM == 3072
    assert config.GEMINI_MODEL
    assert config.VECTOR_DB_DIR.is_dir()
    assert config.FAISS_INDEX_FILE.parent == config.VECTOR_DB_DIR

def test_gemini_model_aliases_and_data_paths():
    assert config.MODEL_NAME == config.GEMINI_MODEL
    assert config.DATA_RAW_DIR == config.DATA_DIR / "raw"
    assert config.DATA_PROCESSED_DIR == config.DATA_DIR / "processed"
