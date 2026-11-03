from __future__ import annotations
import logging
import pytest
import rag.config as config

@pytest.mark.unit
def test_get_logger_reuses_configured_logger_and_handles_file_error(monkeypatch):
    name="coverage.config.logger"
    logger=logging.getLogger(name)
    old_handlers=list(logger.handlers)
    try:
        for handler in old_handlers: logger.removeHandler(handler)
        original_get_logger=logging.getLogger
        monkeypatch.setattr(config.logging,"getLogger",lambda _name=None: logger if _name==name else original_get_logger(_name))
        logger.handlers=[]
        monkeypatch.setattr(config.logging,"FileHandler",lambda *a,**kw: (_ for _ in ()).throw(OSError("readonly")))
        assert config.get_logger(name) is logger
        assert len(logger.handlers)==1
        assert config.get_logger(name) is logger and len(logger.handlers)==1
    finally:
        for handler in list(logger.handlers): logger.removeHandler(handler)
        for handler in old_handlers: logger.addHandler(handler)
