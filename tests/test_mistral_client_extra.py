from __future__ import annotations
from types import SimpleNamespace
import pytest
import rag_test_support  # noqa: F401 - injecte les SDK optionnels avant import des sources
import src.llm.mistral_client as module
from src.llm.mistral_client import MistralClient

@pytest.mark.unit
def test_constructor_validation_and_sdk_init(monkeypatch):
    created = []
    monkeypatch.setattr(module, "Mistral", lambda **kw: created.append(kw) or SimpleNamespace())
    monkeypatch.setattr(module.config, "MISTRAL_API_KEY", "")
    with pytest.raises(ValueError, match="clé API"): MistralClient()
    monkeypatch.setattr(module.config, "MISTRAL_API_KEY", "config-key")
    monkeypatch.setattr(module.config, "MODEL_NAME", "configured")
    client = MistralClient()
    assert client.api_key == "config-key" and client.model == "configured"
    assert created == [{"api_key": "config-key"}]
    monkeypatch.setattr(module.config, "MODEL_NAME", "")
    with pytest.raises(ValueError, match="MODEL_NAME"): MistralClient(api_key="x", model="")

@pytest.mark.unit
def test_generate_messages_parameters_alias_and_blank_response(monkeypatch):
    calls = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    monkeypatch.setattr(module, "Mistral", lambda **kw: SimpleNamespace())
    client = MistralClient("key", "model")
    client.client = SimpleNamespace(chat=SimpleNamespace(complete=lambda **kw: calls.append(kw) or response))
    assert client.generate("Q", "sys", temperature=.8, max_tokens=17, top_p=.6) == "ok"
    assert calls[0]["messages"] == [{"role":"system", "content":"sys"}, {"role":"user", "content":"Q"}]
    assert (calls[0]["temperature"], calls[0]["max_tokens"], calls[0]["top_p"]) == (.8,17,.6)
    response.choices[0].message.content = None
    assert client.generate_response("Q") == ""
    assert MistralClient._messages("Q") == [{"role":"user", "content":"Q"}]

@pytest.mark.unit
def test_generate_and_stream_errors_are_wrapped(monkeypatch):
    monkeypatch.setattr(module, "Mistral", lambda **kw: SimpleNamespace())
    client = MistralClient("key", "model")
    client.client = SimpleNamespace(chat=SimpleNamespace(complete=lambda **kw: (_ for _ in ()).throw(ValueError("api"))))
    with pytest.raises(RuntimeError, match="api"): client.generate("q")
    client.client.chat.stream = lambda **kw: iter([SimpleNamespace(data=SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="x"))])), SimpleNamespace(data=None)])
    assert list(client.generate_stream("q")) == ["x"]
    client.client.chat.stream = lambda **kw: (_ for _ in ()).throw(ValueError("stream"))
    with pytest.raises(RuntimeError, match="stream"): list(client.generate_stream("q"))

@pytest.mark.unit
def test_embeddings_empty_config_and_success(monkeypatch):
    monkeypatch.setattr(module, "Mistral", lambda **kw: SimpleNamespace())
    client = MistralClient("key", "model")
    assert client.embeddings([]) == []
    monkeypatch.setattr(module.config, "EMBEDDING_MODEL", None)
    with pytest.raises(ValueError, match="EMBEDDING_MODEL"): client.embeddings(["x"])
    monkeypatch.setattr(module.config, "EMBEDDING_MODEL", "embed-model")
    calls = []
    client.client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: calls.append(kw) or SimpleNamespace(data=[SimpleNamespace(embedding=[1.,2.]), SimpleNamespace(embedding=[3.,4.])])) )
    assert client.embeddings(["a", "b"]) == [[1.,2.],[3.,4.]]
    assert calls == [{"model":"embed-model", "inputs":["a","b"]}]

@pytest.mark.unit
def test_sdk_missing_ssl_kwargs_and_embedding_errors(monkeypatch):
    monkeypatch.setattr(module,"Mistral",None)
    with pytest.raises(RuntimeError,match="SDK Mistral"): MistralClient("key","model")
    monkeypatch.setattr(module,"Mistral",lambda **kw: SimpleNamespace())
    monkeypatch.setattr(module,"_SSL_BYPASS_ENABLED",False)
    received=[]
    monkeypatch.setattr(module,"Mistral",lambda **kw: received.append(kw) or SimpleNamespace())
    client=MistralClient("key","model",embedding_model="custom")
    assert received==[{"api_key":"key"}]
    client.client=SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(ValueError("embedding"))))
    with pytest.raises(RuntimeError,match="embedding"): client.embeddings(["x"])
    client.embeddings=lambda texts:[texts]
    assert client.generate_embeddings(["x"],batch_size=1)==[["x"]]
    answer=[]
    client.generate=lambda prompt,max_tokens: answer.append((prompt,max_tokens)) or "ok"
    assert client.chat_with_context("Q",["A","B"],max_tokens=20)=={"answer":"ok"}
    assert answer==[("Contexte:\nA\n\nB\nQuestion: Q",20)]

@pytest.mark.unit
def test_httpx_ssl_clients_used_when_bypass_enabled(monkeypatch):
    import ssl
    calls=[]
    monkeypatch.setattr(module,"_SSL_BYPASS_ENABLED",True)
    monkeypatch.setattr(module.httpx,"Client",lambda **kw:calls.append(("sync",kw)) or "c")
    monkeypatch.setattr(module.httpx,"AsyncClient",lambda **kw:calls.append(("async",kw)) or "a")
    monkeypatch.setattr(module,"Mistral",lambda **kw: calls.append(("sdk",kw)) or SimpleNamespace())
    client=MistralClient("key","model")
    assert calls==[("sync",{"verify":False}),("async",{"verify":False}),("sdk",{"api_key":"key","client":"c","async_client":"a"})]
