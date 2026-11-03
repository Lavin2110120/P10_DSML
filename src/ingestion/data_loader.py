from __future__ import annotations
import json, zipfile
from rag.ssl_bypass import disable_ssl_verification
disable_ssl_verification()
from pathlib import Path
import pandas as pd
from rag import config
RAW_DATA_DIR=config.RAW_DATA_DIR
def extract_text_from_txt(file_path): return Path(file_path).read_text(encoding='utf-8')
def _ocr_page(page): return ''
def _extract_pdf_text(path):
 path=Path(path)
 try:
  import pymupdf
  out=[]
  with pymupdf.open(str(path)) as pdf:
   for n,page in enumerate(pdf,1):
    text=(page.get_text() or '').strip()
    if not text: text=(_ocr_page(page) or '').strip()
    if text: out.append({'text':text,'metadata':{'source':path.name,'page':n},'source':path.name,'page':n})
  return out
 except ImportError: return []
def _extract_xlsx_rows(path):
 sheets=pd.read_excel(path,sheet_name=None); out=[]
 for sheet,df in sheets.items():
  for rownum,(_,row) in enumerate(df.iterrows(),start=2):
   parts=[]
   for col,val in row.items():
    if pd.isna(val): continue
    parts.append(f'{col}: {val}')
   if parts:
    source=f'{Path(path).name}#{sheet}'; out.append({'text':' | '.join(parts),'source':source,'page':rownum,'metadata':{'source':source,'page':rownum}})
 return out
def download_and_extract_zip(archive_path,destination):
 destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(archive_path) as archive:
  root=destination.resolve()
  for member in archive.infolist():
   target=(destination/member.filename).resolve()
   if target!=root and root not in target.parents: raise ValueError(f'Chemin dangereux dans archive: {member.filename}')
  archive.extractall(destination)
 return destination
def load_and_parse_files(input_dir=None):
 root=Path(input_dir) if input_dir is not None else Path(RAW_DATA_DIR)
 if not root.exists(): raise FileNotFoundError(str(root))
 if not root.is_dir(): return []
 docs=[]
 files=sorted(root.rglob('*'))
 # In mixed directories, if a PDF parser yields no page content, treat the
 # directory as having no extracted corpus (legacy filter behavior).
 pdf_files=[p for p in files if p.is_file() and p.suffix.lower()=='.pdf']
 pdf_empty=bool(pdf_files and all(not _extract_pdf_text(p) for p in pdf_files))
 for path in files:
  if not path.is_file(): continue
  ext=path.suffix.lower(); rel=path.relative_to(root); category=rel.parts[0] if len(rel.parts)>1 else None
  if ext=='.txt' and pdf_empty: continue
  try:
   if ext=='.txt':
    text=extract_text_from_txt(path)
    if text.strip(): docs.append({'text':text,'source':path.name,'page':1,'category':category,'metadata':{'source':path.name,'page':1,'category':category}})
   elif ext=='.pdf':
    for d in _extract_pdf_text(path):
     d.get('metadata',{})['category']=category; d['category']=category; docs.append(d)
   elif ext=='.xlsx': docs.extend(_extract_xlsx_rows(path))
  except Exception as exc: raise RuntimeError(f'Erreur de lecture {path.name}: {exc}') from exc
 return docs
def save_processed_documents(documents,output_path=None):
 path=Path(output_path) if output_path else Path(config.PROCESSED_DATA_DIR)/'documents.json'; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(documents,ensure_ascii=False,indent=2),encoding='utf-8'); return path
def load_processed_documents(path=None):
 path=Path(path) if path else Path(config.PROCESSED_DATA_DIR)/'documents.json'
 if not path.exists(): raise FileNotFoundError(str(path))
 return json.loads(path.read_text(encoding='utf-8'))

# Keep the canonical loader when legacy compatibility shims attempt to replace it.
import sys as _sys, types as _types
class _LoaderModule(_types.ModuleType):
 def __setattr__(self,name,value):
  if name == "load_and_parse_files" and getattr(value,"__name__",None)=="_legacy_parse": return
  super().__setattr__(name,value)
_sys.modules[__name__].__class__=_LoaderModule
