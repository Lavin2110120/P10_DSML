"""Offline tests for optional and idempotent Logfire instrumentation."""
from contextlib import contextmanager
from types import SimpleNamespace
from src import observability

def test_noop_when_logfire_missing(monkeypatch):
    monkeypatch.setattr(observability, "_logfire", None)
    monkeypatch.setattr(observability, "_configured", False)
    observability.configure(); observability.configure()
    with observability.span("safe", top_k=3) as trace:
        assert trace is None

def test_configure_once_and_span_attributes(monkeypatch):
    calls=[]; spans=[]
    @contextmanager
    def span(name, **attrs):
        spans.append((name, attrs)); yield SimpleNamespace(set_attribute=lambda *a: None)
    fake=SimpleNamespace(configure=lambda **kw: calls.append(kw), span=span)
    monkeypatch.setattr(observability, "_logfire", fake)
    monkeypatch.setattr(observability, "_configured", False)
    observability.configure(); observability.configure()
    with observability.span("rag.search", top_k=4): pass
    assert len(calls)==1 and calls[0]["send_to_logfire"] == "if-token-present"
    assert spans == [("rag.search", {"top_k":4})]
