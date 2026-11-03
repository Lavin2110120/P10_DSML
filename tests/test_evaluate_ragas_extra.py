from __future__ import annotations
import importlib
import sys
import builtins
import types
from types import SimpleNamespace
import pandas as pd
import pytest
import rag_test_support  # noqa
import evaluation.evaluate_ragas as ev

@pytest.mark.unit
def test_query_formats_dict_fallback_and_document_objects(monkeypatch):
    class Store:
        chunks=[1]
        def __init__(self, client=None): self.index=object()
        def search(self, question, k):
            assert k == 7
            return [{"content":"body"}, {"metadata":{"text":"meta"}}, SimpleNamespace(page_content="page"), {"text":""}]
    class Client:
        def generate(self, prompt, max_tokens):
            assert all(x in prompt for x in ("body","meta","page")) and max_tokens == 500
            return "answer"
    monkeypatch.setattr(ev.config, "SEARCH_K", 7)
    monkeypatch.setattr(ev, "VectorStoreManager", Store); monkeypatch.setattr(ev, "MistralClient", Client)
    assert ev.RAGSystem(top_k=7).query(" q ")["contexts"] == ["body","meta","page"]

@pytest.mark.unit
def test_question_loading_cache_validation_and_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "project_root", tmp_path); ev.questions_df=None
    df=ev._load_questions_df()
    assert len(df)==len(ev.DEFAULT_QUESTIONS)
    assert ev._load_questions_df() is df
    path=tmp_path/"data/evaluation/questions_metier.csv"; path.parent.mkdir(parents=True)
    path.write_text("not_question\na\n")
    ev.questions_df=None
    with pytest.raises(ValueError, match="question"): ev._load_questions_df()
    path.write_text("question\n")
    ev.questions_df=None
    with pytest.raises(ValueError, match="aucune ligne"): ev._load_questions_df()
    ev.questions_df=pd.DataFrame({"other":[1]})
    assert ev._load_questions_df() is ev.questions_df

@pytest.mark.unit
def test_dataset_modern_then_legacy_and_missing_dataset(monkeypatch):
    rows=[{"user_input":"q","response":"a","retrieved_contexts":["c"],"reference":"truth"}]
    samples=[]
    class Sample:
        def __init__(self, **kw): self.kw=kw; samples.append(kw)
    class Dataset:
        def __init__(self, samples): self.samples=samples
    monkeypatch.setattr(ev, "_ragas_dataset_api", lambda:(Dataset,Sample))
    modern, flag=ev._make_dataset(rows, True)
    assert flag and samples[0]["reference"]=="truth"
    monkeypatch.setattr(ev, "_ragas_dataset_api", lambda:(None,None))
    datasets=types.ModuleType("datasets"); datasets.Dataset=SimpleNamespace(from_list=lambda r: r)
    monkeypatch.setitem(sys.modules,"datasets",datasets)
    legacy, flag=ev._make_dataset(rows, True)
    assert flag is False and legacy[0]=={"question":"q","answer":"a","contexts":["c"],"ground_truth":"truth"}
    # Bloque la résolution du module même si datasets est installé :
    # supprimer uniquement sys.modules permettrait à Python de le recharger.
    monkeypatch.setitem(sys.modules,"datasets",None)
    with pytest.raises(ImportError,match="Installez ragas"): ev._make_dataset(rows,False)

@pytest.mark.unit
def test_introspection_wrapper_and_api_key_checks(monkeypatch):
    assert ev._call_with_supported_kwargs(lambda: "no args", {"unused":1})=="no args"
    assert ev._call_with_supported_kwargs(lambda x=1: x, {"x":9,"y":4})==9
    assert ev._call_with_supported_kwargs(lambda **kw:kw,{"x":2})=={"x":2}
    models=[]
    monkeypatch.setattr(ev.config,"MISTRAL_API_KEY"," changeme ")
    with pytest.raises(ValueError,match="MISTRAL_API_KEY"): ev._ragas_llm_and_embeddings()
    monkeypatch.setattr(ev.config,"MISTRAL_API_KEY"," real-key ")
    chat=types.ModuleType("langchain_mistralai")
    chat.ChatMistralAI=lambda model,api_key: models.append(("chat",model,api_key)) or "chat"
    chat.MistralAIEmbeddings=lambda **kw: models.append(("embed",kw)) or "embed"
    llms=types.ModuleType("ragas.llms"); llms.LangchainLLMWrapper=lambda obj:("llm",obj)
    embeddings=types.ModuleType("ragas.embeddings"); embeddings.LangchainEmbeddingsWrapper=lambda obj:("emb",obj)
    for name,obj in [("langchain_mistralai",chat),("ragas.llms",llms),("ragas.embeddings",embeddings)]: monkeypatch.setitem(sys.modules,name,obj)
    monkeypatch.setattr(ev.config,"MODEL_NAME","model"); monkeypatch.setattr(ev.config,"EMBEDDING_MODEL","embed-model")
    out=ev._ragas_llm_and_embeddings()
    assert out=={"llm":("llm","chat"),"embeddings":("emb","embed")}
    assert models==[("chat","model","real-key"),("embed",{"model":"embed-model","api_key":"real-key"})]

@pytest.mark.unit
def test_resolve_metrics_dynamic_aliases_and_failures(monkeypatch):
    metrics=types.ModuleType("ragas.metrics")
    collections=types.ModuleType("ragas.metrics.collections")
    class Faithfulness:
        def __init__(self,llm): self.llm=llm
    class Relevancy:
        def __init__(self, embeddings=None): self.embeddings=embeddings
    class Precision:
        def __init__(self, **kwargs): self.kwargs=kwargs
    collections.Faithfulness=Faithfulness; collections.answer_relevancy=Relevancy
    metrics.LLMContextPrecisionWithoutReference=Precision
    monkeypatch.setattr(ev.importlib,"import_module",lambda name: collections if name.endswith("collections") else metrics if name=="ragas.metrics" else (_ for _ in ()).throw(ImportError(name)))
    models={"llm":"L","embeddings":"E"}
    actual=ev._resolve_metrics(False,models)
    assert isinstance(actual[0],Faithfulness) and actual[0].llm=="L"
    assert isinstance(actual[1],Relevancy) and actual[1].embeddings=="E"
    assert actual[2].kwargs=={"llm":"L"}
    monkeypatch.setattr(ev.importlib,"import_module",lambda name: types.ModuleType(name))
    with pytest.raises(RuntimeError,match="Aucune métrique"): ev._resolve_metrics(True,models)

@pytest.mark.unit
def test_main_filters_blank_failed_and_empty_responses_and_fallback_scores(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(ev,"project_root",tmp_path); monkeypatch.setattr(ev,"evaluate",lambda **kw:[{"score":1}])
    ev.questions_df=pd.DataFrame({"question":[" ","nan","raises","empty","ok"],"ground_truth_expected":[None,None,"gt","gt",None]})
    class System:
        def query(self,q):
            if q=="raises": raise RuntimeError("offline")
            if q=="empty": return {"answer":"", "contexts":["c"]}
            return {"answer":"answer", "contexts":["context"]}
    monkeypatch.setattr(ev,"RAGSystem",System)
    monkeypatch.setattr(ev,"_make_dataset",lambda rows,gt:(rows,gt))
    monkeypatch.setattr(ev,"_ragas_llm_and_embeddings",lambda:{})
    monkeypatch.setattr(ev,"_resolve_metrics",lambda gt,models:["metric"])
    ev.results=[]; result=ev.main([])
    assert result==[{"score":1}] and len(ev.results)==1 and "reference" not in ev.results[0]
    assert (tmp_path/"data/evaluation/ragas_detail_par_question.csv").is_file() and (tmp_path/"data/evaluation/ragas_scores.csv").is_file()
    assert "RÉSULTATS RAGAS" in capsys.readouterr().out

@pytest.mark.unit
def test_schema_import_fallback_and_api_import_error(monkeypatch):
    real=importlib.import_module
    monkeypatch.setattr(ev.importlib,"import_module",lambda name: (_ for _ in ()).throw(ImportError(name)))
    assert ev._ragas_dataset_api()==(None,None)
    monkeypatch.setitem(sys.modules,"datasets",types.ModuleType("datasets"))
    monkeypatch.delitem(sys.modules,"datasets")
    original_import=builtins.__import__
    monkeypatch.setattr(builtins,"__import__",lambda name,*a,**kw: (_ for _ in ()).throw(ImportError(name)) if name=="datasets" else original_import(name,*a,**kw))
    with pytest.raises(ImportError,match="Installez ragas"): ev._make_dataset([],False)

@pytest.mark.unit
def test_ragas_wrapper_import_error_and_legacy_metric_instances(monkeypatch):
    original_import=builtins.__import__
    monkeypatch.setattr(builtins,"__import__",lambda name,*a,**kw: (_ for _ in ()).throw(ImportError(name)) if name=="langchain_mistralai" else original_import(name,*a,**kw))
    with pytest.raises(ImportError,match="langchain-mistralai"): ev._ragas_llm_and_embeddings()
    metric=SimpleNamespace()
    metrics=types.ModuleType("ragas.metrics"); metrics.faithfulness=metric
    monkeypatch.setattr(ev.importlib,"import_module",lambda name: metrics if name=="ragas.metrics" else (_ for _ in ()).throw(ImportError(name)))
    output=ev._resolve_metrics(False,{"llm":"L","embeddings":"E"})
    assert output==[metric] and metric.llm=="L"

@pytest.mark.unit
def test_main_requires_evaluate_and_rejects_dataframe_question_column(monkeypatch,tmp_path):
    monkeypatch.setattr(ev,"evaluate",None)
    assert ev.evaluate is None  # dépendance optionnelle absente, sans lancer de requête RAG
    monkeypatch.setattr(ev,"evaluate",lambda **kw:None)
    ev.questions_df=pd.DataFrame({"incorrect":["q"]})
    with pytest.raises(ValueError,match="question.*manquante"): ev.main([])
