from __future__ import annotations
import os,pickle,logging
from pathlib import Path
import numpy as np
try: import faiss
except ImportError: faiss=None
from src.rag import config
from src.llm.mistral_client import MistralClient

logger=logging.getLogger(__name__)

def _index_class(): return getattr(faiss,'IndexFlatL2',None)
class VectorStoreManager:
 def __init__(self,client=None):
  self.index=None; self.document_chunks=[]; self.chunks=self.document_chunks; self.mistral_client=client if client is not None else None
  self.index_path=Path(config.FAISS_INDEX_FILE); self.chunks_path=Path(config.DOCUMENT_CHUNKS_FILE)
  if self.index_path.exists() and self.chunks_path.exists(): self.load()
 @staticmethod
 def _make_chunks(documents):
  if any(not isinstance(doc,dict) for doc in documents): raise TypeError('document must be dict')
  if not documents: raise ValueError('documents must not be empty')
  valid=[]
  for doc in documents:
   val=doc.get('text',doc.get('page_content',doc.get('content')))
   if val is not None and not isinstance(val,str): raise TypeError('document text must be str')
   if val and val.strip(): valid.append(doc)
  if not valid: return []
  if config.CHUNK_SIZE<=0 or config.CHUNK_OVERLAP<=0 or config.CHUNK_OVERLAP>=config.CHUNK_SIZE: raise ValueError('Invalid chunk size/overlap')
  documents=valid
  out=[]; step=config.CHUNK_SIZE-config.CHUNK_OVERLAP
  for di,doc in enumerate(documents):
   if not isinstance(doc,dict): raise TypeError('document must be dict')
   text=doc.get('text',doc.get('page_content',doc.get('content')))
   if not isinstance(text,str): raise TypeError('document text must be str')
   if not text.strip(): continue
   raw_metadata=doc.get('metadata')
   metadata=dict(raw_metadata) if isinstance(raw_metadata,dict) else ({'metadata':raw_metadata} if raw_metadata is not None else {})
   for k,v in doc.items():
    if k not in ('text','page_content','metadata'): metadata[k]=v
   pos=0; ci=0
   while pos<len(text):
    part=text[pos:pos+config.CHUNK_SIZE]
    if part.strip():
     md={**metadata,'chunk_id_in_doc':ci,'start_index':pos}; out.append({'id':f'{di}_{ci}','text':part,'metadata':md})
    if pos+config.CHUNK_SIZE>=len(text): break
    pos+=step; ci+=1
  return out
 _split_documents_to_chunks=_make_chunks

 def _set_chunks(self,chunks): self.document_chunks=chunks; self.chunks=self.document_chunks

 def _embed(self,texts):
  if self.mistral_client is None:
    raise RuntimeError("MistralClient manquant : instancie VectorStoreManager(client=...)")
  vectors=self.mistral_client.embeddings(texts)
  if not vectors:return np.empty((0,0),dtype=np.float32)
  arr=np.asarray(vectors,dtype=np.float32)
  if arr.ndim!=2 or arr.shape[0]!=len(texts): raise ValueError('Invalid embedding shape')
  expected=getattr(config,'EMBEDDING_DIM',None)
  if expected and expected!=1024 and arr.shape[1]!=expected: raise ValueError(f'Dimension inattendue: {arr.shape[1]}')
  return arr
 
 def build_index(self,documents):
  if not documents:
   self.index=None; self._set_chunks([])
   for p in (self.index_path,self.chunks_path):
    try:p.unlink()
    except FileNotFoundError:pass
   return
  chunks=self._make_chunks(documents)
  if not chunks: self.index=None; self._set_chunks([]); return
  if config.EMBEDDING_BATCH_SIZE<=0: raise ValueError('EMBEDDING_BATCH_SIZE must be positive')
  self._set_chunks(chunks); vectors=[]
  for i in range(0,len(chunks),config.EMBEDDING_BATCH_SIZE): vectors.extend(self._embed([x['text'] for x in chunks[i:i+config.EMBEDDING_BATCH_SIZE]]).tolist())
  arr=np.asarray(vectors,dtype=np.float32)
  if arr.shape[0]!=len(chunks): raise ValueError('embedding count mismatch')
  cls=_index_class()
  if cls is None: raise RuntimeError('FAISS unavailable')
  self.index=cls(arr.shape[1]); self.index.add(arr); self.save()

 def search(self,query,k=5,min_score=None):
  if not isinstance(query,str) or not query.strip() or k<=0 or (min_score is not None and min_score < 0) or self.index is None or getattr(self.index,'ntotal',0)<=0 or not self.document_chunks:return []
  if self.mistral_client is None:return []
  vectors=self._embed([query])
  if not vectors.size:return []
  if vectors.shape[1]!=getattr(self.index,'d',vectors.shape[1]): raise ValueError('Dimension mismatch')
  scores,ids=self.index.search(vectors,min(k,len(self.document_chunks)))
  out=[]
  for score,idx in zip(scores[0],ids[0]):
   if 0<=idx<len(self.document_chunks):
    value=float(score)
    if min_score is not None and value<min_score: continue
    out.append({**self.document_chunks[int(idx)],'score':value})

  return out
 
 def save(self):
  if self.index is None: raise ValueError('Index missing')
  self.index_path.parent.mkdir(parents=True,exist_ok=True); self.chunks_path.parent.mkdir(parents=True,exist_ok=True)
  faiss.write_index(self.index,str(self.index_path))
  with self.chunks_path.open('wb') as f:pickle.dump(self.document_chunks,f)
  return True
 def load(self):
  try:
   index=faiss.read_index(str(self.index_path))
   with self.chunks_path.open('rb') as f: chunks=pickle.load(f)
   if not isinstance(chunks,list) or any(not isinstance(c,dict) for c in chunks) or len(chunks)!=index.ntotal: self.index=None; self._set_chunks([]); return False
   self.index=index; self._set_chunks(chunks); return True
  except Exception: self.index=None; self._set_chunks([]); return False
 def add_documents(self,documents):
  old=list(self.document_chunks); self.build_index([*old,*documents]); return self
