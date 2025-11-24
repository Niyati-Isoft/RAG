# rag_core.py
# -------------------------------------------------------------------
# Core Multimedia Token-based RAG (backend-only, no Streamlit / no KG)
#
# Responsibilities:
#   1) Chunking with FLAN-T5 tokenizer
#   2) File loaders (PDF/DOCX/PPTX/CSV/XLSX/TXT/MD/HTML/Images/Audio/Video/Code)
#   3) Website crawling + linked files
#   4) Metadata enrichment (doc_id, chunk_id, topic, source_domain, etc.)
#   5) Vector DB build/load (FAISS + Weaviate)
#   6) Helper to extract stored docs from FAISS / Weaviate
#   7) RAG pipeline: retrieve → build context → draft → optional polish
#
# Things **NOT** included:
#   - Router / Orchestrator
#   - Health / General agents
#   - Neo4j / KG / Graphs / PyVis
#   - Streamlit UI
#
# Agents will call:
#   from rag_core import (
#       ingest_files_from_dir,
#       crawl_and_ingest_web,
#       build_faiss_index,
#       load_faiss_index,
#       build_weaviate_index,
#       load_weaviate_index,
#       make_retriever,
#       run_rag_pipeline,
#       fetch_docs_from_weaviate,
#       faiss_docs_from_store,
#   )
# -------------------------------------------------------------------

import os
import re
import io
import json
import math
import time
import shutil
import tempfile
import urllib.parse
from pathlib import Path
from typing import List, Dict, Iterable, Optional, Tuple, Set
from collections import defaultdict, Counter

import numpy as np
import requests

# LangChain / HF
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    AutoTokenizer as HFAutoTokenizer,
)

# Optional deps (loaded lazily)
def _optional_import(name, alias=None):
    try:
        mod = __import__(name)
        return mod if alias is None else __import__(name, fromlist=[alias])
    except Exception:
        return None

pdfplumber  = _optional_import("pdfplumber")
pypdf       = _optional_import("pypdf")
docx        = _optional_import("docx")
pptx        = _optional_import("pptx")
pytesseract = _optional_import("pytesseract")
bs4         = _optional_import("bs4", "BeautifulSoup")
trafilatura = _optional_import("trafilatura")
whisper     = _optional_import("whisper")
moviepy     = _optional_import("moviepy")
weaviate    = _optional_import("weaviate")

# -------------------------------------------------------------------
# 0. Embeddings & Local LLM
# -------------------------------------------------------------------

# Embeddings (global, shared)
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True},
)

# Local FLAN-T5 cache
_tokenizer_cache = {}
_llm_pipeline_cache = {}


def get_tokenizer_and_llm(model_name: str):
    """
    Build / cache FLAN-T5 tokenizer + text2text-generation pipeline.
    Used when llm_engine == 'flan'.
    """
    if model_name in _tokenizer_cache and model_name in _llm_pipeline_cache:
        return _tokenizer_cache[model_name], _llm_pipeline_cache[model_name]

    tok = AutoTokenizer.from_pretrained(model_name)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")

    gen_pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=384,
        temperature=0.4,
        num_beams=3,
        repetition_penalty=1.05,
        device=0 if torch.cuda.is_available() else -1,
    )

    llm = HuggingFacePipeline(pipeline=gen_pipe)
    _tokenizer_cache[model_name] = tok
    _llm_pipeline_cache[model_name] = llm
    return tok, llm


# For token counting (embedder vs FLAN)
_mini_tok_cache = None


def get_mini_tokenizer():
    global _mini_tok_cache
    if _mini_tok_cache is None:
        _mini_tok_cache = HFAutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    return _mini_tok_cache


# -------------------------------------------------------------------
# 1. Chunking helpers (FLAN tokenizer)
# -------------------------------------------------------------------

# Default values (your UI can override them)
DEFAULT_CHUNK_SIZE_TOKENS = 200
DEFAULT_CHUNK_OVERLAP_TOKENS = 40

# Default local FLAN model name
DEFAULT_FLAN_MODEL_NAME = "google/flan-t5-base"

# Module-global tokenizer (for chunking) – uses FLAN
chunk_tokenizer, _ = get_tokenizer_and_llm(DEFAULT_FLAN_MODEL_NAME)


def chunk_by_hf_tokens(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> List[str]:
    """
    Chunk text using the FLAN tokenizer.
    """
    if not text or not text.strip():
        return []
    ids = chunk_tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(ids), step):
        piece = ids[start : start + chunk_size]
        if not piece:
            break
        chunks.append(chunk_tokenizer.decode(piece, skip_special_tokens=True))
        if start + chunk_size >= len(ids):
            break
    return chunks


# -------------------------------------------------------------------
# 2. Document construction + metadata enrichment
# -------------------------------------------------------------------

import hashlib


def _stable_id(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def to_docs(chunks: Iterable[str], base_meta: Dict) -> List[Document]:
    """
    Convert raw chunk strings → LangChain Documents, adding:
        doc_id (stable hash of source/url)
        chunk_id (increment)
    """
    anchor = (
        base_meta.get("url")
        or base_meta.get("source")
        or json.dumps(base_meta, sort_keys=True)
    )
    doc_id = _stable_id(anchor)

    docs = []
    for i, c in enumerate([x for x in chunks if x and str(x).strip()], start=1):
        meta = base_meta.copy()
        meta["doc_id"] = doc_id
        meta["chunk_id"] = f"{doc_id}:{i}"
        docs.append(Document(page_content=c.strip(), metadata=meta))
    return docs



import tldextract


def _domain_of(url_or_path: str) -> Optional[str]:
    u = str(url_or_path or "")
    if "://" not in u:
        return None
    ext = tldextract.extract(u)
    dom = ".".join([p for p in [ext.domain, ext.suffix] if p])
    sub = ext.subdomain
    return f"{sub}.{dom}".strip(".") if sub else dom


def enrich_docs(
    docs: List[Document],
    source_type: str,
    source_url: Optional[str] = None,
    file_path: Optional[str] = None,
) -> List[Document]:
    """
    Enrich docs with metadata:
      - doc_id, chunk_id (if not already set)
      - source_type (file/web)
      - source_url / file_name
      - embedding_model, created_at
      - source_domain

    No automatic health detection / topic tagging here.
    Same ingestion for both HEALTH and GENERAL agents.
    """
    import uuid

    doc_id = str(uuid.uuid4())
    file_name = Path(file_path).name if file_path else None
    source_domain = _domain_of(
        source_url or (docs[0].metadata.get("url") if docs else None)
    )

    for i, d in enumerate(docs, 1):
        m = dict(d.metadata or {})
        # Respect existing doc_id / chunk_id if already set
        m.setdefault("doc_id", doc_id)
        m.setdefault("chunk_id", f"{doc_id}-{i:04d}")
        m["source_type"] = source_type
        if source_url:
            m["source_url"] = source_url
        if file_name:
            m["file_name"] = file_name
        m["embedding_model"] = EMBED_MODEL_NAME
        m["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if source_domain:
            m["source_domain"] = source_domain
        # 🔴 no auto m["topic"] assignment here
        d.metadata = m
    return docs


# -------------------------------------------------------------------
# 3. File loaders (non-web)
# -------------------------------------------------------------------

from PIL import Image

SUPPORT_NONWEB = {
    ".pdf",
    ".docx",
    ".pptx",
    ".csv",
    ".xlsx",
    ".txt",
    ".md",
    ".html",
    ".htm",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".mp3",
    ".wav",
    ".m4a",
    ".mp4",
    ".mkv",
    ".mov",
    ".py",
    ".js",
    ".java",
    ".cpp",
}


def load_pdf(path: str) -> List[Document]:
    docs = []
    if pdfplumber:
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs += to_docs(
                            chunk_by_hf_tokens(text),
                            {"source": path, "modality": "document", "page": i},
                        )
            return docs
        except Exception:
            pass
    if pypdf:
        try:
            reader = pypdf.PdfReader(path)
            for i, pg in enumerate(reader.pages, start=1):
                text = pg.extract_text() or ""
                if text.strip():
                    docs += to_docs(
                        chunk_by_hf_tokens(text),
                        {"source": path, "modality": "document", "page": i},
                    )
        except Exception:
            pass
    return docs


def load_docx(path: str) -> List[Document]:
    if not docx:
        return []
    d = docx.Document(path)
    text = "\n".join(p.text for p in d.paragraphs)
    return to_docs(
        chunk_by_hf_tokens(text), {"source": path, "modality": "document"}
    )


def load_pptx(path: str) -> List[Document]:
    if not pptx:
        return []
    prs = pptx.Presentation(path)
    docs = []
    for si, slide in enumerate(prs.slides, start=1):
        texts = []
        for shp in slide.shapes:
            if hasattr(shp, "text"):
                t = (shp.text or "").strip()
                if t:
                    texts.append(t)
        text = "\n".join(texts)
        if text.strip():
            docs += to_docs(
                chunk_by_hf_tokens(text),
                {"source": path, "modality": "document", "slide": si},
            )
    return docs


def load_csv(path: str) -> List[Document]:
    import pandas as pd

    df = pd.read_csv(path)
    lines = [
        f"{i}: " + "; ".join(f"{c}={df.loc[i, c]}" for c in df.columns)
        for i in range(len(df))
    ]
    text = "\n".join(lines)
    return to_docs(chunk_by_hf_tokens(text), {"source": path, "modality": "table"})


def load_xlsx(path: str) -> List[Document]:
    import pandas as pd

    docs = []
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        lines = [
            f"{i}: " + "; ".join(f"{c}={df.loc[i, c]}" for c in df.columns)
            for i in range(len(df))
        ]
        text = "\n".join(lines)
        if text.strip():
            docs += to_docs(
                chunk_by_hf_tokens(text),
                {"source": path, "modality": "table", "sheet": sheet},
            )
    return docs


def load_textlike(path: str) -> List[Document]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return to_docs(chunk_by_hf_tokens(text), {"source": path, "modality": "text"})


def load_html(path: str) -> List[Document]:
    if not bs4:
        return []
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = bs4.BeautifulSoup(html, "html.parser")
    for t in soup(
        ["script", "style", "noscript", "iframe", "svg", "header", "footer", "nav"]
    ):
        t.decompose()
    text = soup.get_text("\n")
    text = "\n".join(line.strip() for line in text.splitlines())
    text = "\n".join([ln for ln in text.split("\n") if ln])
    if not text.strip():
        return []
    return to_docs(
        chunk_by_hf_tokens(text),
        {"source": path, "modality": "text", "format": "html"},
    )


def load_image(path: str) -> List[Document]:
    if not pytesseract:
        print("pytesseract not installed; skipping OCR for images.")
        return []
    img = Image.open(path)
    text = pytesseract.image_to_string(img)
    if text.strip():
        return to_docs(
            chunk_by_hf_tokens(text), {"source": path, "modality": "image"}
        )
    return []


_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        if not whisper:
            print("whisper not installed; skipping audio/video transcription.")
            return None
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def load_audio(path: str) -> List[Document]:
    model = _get_whisper()
    if model is None:
        return []
    result = model.transcribe(path, verbose=False)
    docs = []
    for seg in result.get("segments", []):
        t = (seg.get("text", "") or "").strip()
        if not t:
            continue
        start = round(seg.get("start", 0), 2)
        end = round(seg.get("end", 0), 2)
        for c in chunk_by_hf_tokens(t):
            docs.append(
                Document(
                    page_content=c,
                    metadata={
                        "source": path,
                        "modality": "audio",
                        "start_sec": start,
                        "end_sec": end,
                    },
                )
            )
    return docs


def load_video(path: str) -> List[Document]:
    if not moviepy:
        print("moviepy not installed; skipping audio extraction from video.")
        return []
    tmp_wav = None
    try:
        clip = moviepy.editor.VideoFileClip(path)
        tmp_wav = str(
            Path(tempfile.gettempdir()) / f"tmp_{int(time.time())}.wav"
        )
        clip.audio.write_audiofile(tmp_wav, verbose=False, logger=None)
        return load_audio(tmp_wav)
    except Exception:
        return []
    finally:
        try:
            if tmp_wav and Path(tmp_wav).exists():
                Path(tmp_wav).unlink()
        except Exception:
            pass


def load_any_nonweb(path: str) -> List[Document]:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    if ext == ".pptx":
        return load_pptx(path)
    if ext == ".csv":
        return load_csv(path)
    if ext == ".xlsx":
        return load_xlsx(path)
    if ext in {".txt", ".md"}:
        return load_textlike(path)
    if ext in {".html", ".htm"}:
        return load_html(path)
    if ext in {".jpg", ".jpeg", ".png", ".webp"}:
        return load_image(path)
    if ext in {".mp3", ".wav", ".m4a"}:
        return load_audio(path)
    if ext in {".mp4", ".mkv", ".mov"}:
        return load_video(path)
    if ext in {".py", ".js", ".java", ".cpp"}:
        return load_textlike(path)
    raise ValueError(f"Unsupported file type: {ext}")


# -------------------------------------------------------------------
# 4. Website crawler (pages + linked files)
# -------------------------------------------------------------------

REQUEST_HEADERS = {"User-Agent": "RAG-Core/1.0"}
REQUEST_TIMEOUT = 20
CRAWL_SLEEP_S = 0.25
MAX_PAGES = 20
MAX_FILES = 10
SAME_HOST_ONLY = True
ALLOWED_FILE_EXTS = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".md"}

HEALTH_ALLOWLIST = set()  # unused by default; can be used to restrict domains


def _normalize_url(url: str) -> str:
    try:
        u = urllib.parse.urlsplit(url)
        u = u._replace(fragment="")
        return urllib.parse.urlunsplit(u)
    except Exception:
        return url


def _abs_url(base: str, link: str) -> str:
    return urllib.parse.urljoin(base, link)


def _same_host(u1: str, u2: str) -> bool:
    return urllib.parse.urlsplit(u1).netloc.lower() == urllib.parse.urlsplit(
        u2
    ).netloc.lower()


def _is_text_html(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    return "text/html" in ctype or "application/xhtml" in ctype


def _visible_text_from_html(html: str) -> str:
    if not bs4:
        return ""
    soup = bs4.BeautifulSoup(html, "html.parser")
    for tag in soup(
        ["script", "style", "noscript", "header", "footer", "nav", "iframe"]
    ):
        tag.decompose()
    text = soup.get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_text(url: str, html: Optional[str] = None) -> str:
    if trafilatura:
        try:
            if html is None:
                downloaded = trafilatura.fetch_url(url)
                txt = trafilatura.extract(downloaded) if downloaded else None
            else:
                txt = trafilatura.extract(html)
        except Exception:
            txt = None
        if txt:
            return txt.strip()
    return _visible_text_from_html(html or "")


def _gather_links(base_url: str, html: str) -> Tuple[Set[str], Set[str]]:
    if not bs4:
        return set(), set()
    soup = bs4.BeautifulSoup(html, "html.parser")
    page_links, file_links = set(), set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        u = _normalize_url(
            _abs_url(base_url, href.split("#", 1)[0])
        )
        ext = Path(urllib.parse.urlsplit(u).path).suffix.lower()
        if ext in ALLOWED_FILE_EXTS:
            file_links.add(u)
        else:
            page_links.add(u)
    return page_links, file_links


def _download_to_tmp(url: str, tmpdir: Path) -> Optional[Path]:
    try:
        r = requests.get(
            url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS, stream=True
        )
        if r.status_code != 200:
            return None
        name = Path(urllib.parse.urlsplit(url).path).name or "download"
        if "." not in name:
            name += ".bin"
        out = tmpdir / name
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)
        return out
    except Exception:
        return None


def _token_chunk_to_docs(text: str, url: str) -> List[Document]:
    if not text:
        return []
    return [
        Document(
            page_content=c,
            metadata={"url": url, "source": url, "modality": "web"},
        )
        for c in chunk_by_hf_tokens(text)
    ]


def crawl_and_ingest_web(
    roots: List[str],
    same_host_only: bool = SAME_HOST_ONLY,
    max_pages: int = MAX_PAGES,
    max_files: int = MAX_FILES,
    sleep_s: float = CRAWL_SLEEP_S,
) -> Tuple[List[Document], List[dict]]:
    """
    Crawl website roots:
      - fetch pages
      - extract text → token-chunks → Documents
      - download linked files (pdf/docx/…) and load them
    Returns: (docs, crawl_log)
      docs: list[Document]
      crawl_log: list of {type: start/page/file, ...} for UI display if needed
    """
    all_docs: List[Document] = []
    events: List[dict] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="rag_web_"))

    try:
        for root in roots:
            if not root:
                continue
            root = _normalize_url(root.strip())
            events.append({"type": "start", "url": root})

            seen_pages, queue = set(), [root]
            pages_fetched, files_fetched = 0, 0

            while queue and pages_fetched < max_pages:
                url = queue.pop(0)

                # Optional: health-domain allowlist
                dom = _domain_of(url)
                if dom and HEALTH_ALLOWLIST and all(
                    allowed not in dom for allowed in HEALTH_ALLOWLIST
                ):
                    continue

                if url in seen_pages:
                    continue
                seen_pages.add(url)

                if same_host_only and not _same_host(root, url):
                    continue

                try:
                    resp = requests.get(
                        url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS
                    )
                except Exception:
                    continue
                if resp.status_code != 200 or not _is_text_html(resp):
                    continue

                text = _extract_text(url, resp.text)
                page_docs = _token_chunk_to_docs(text, url)
                all_docs.extend(page_docs)
                pages_fetched += 1
                events.append(
                    {
                        "type": "page",
                        "idx": pages_fetched,
                        "url": url,
                        "chunks": len(page_docs),
                    }
                )

                page_links, file_links = _gather_links(url, resp.text)
                for nxt in page_links:
                    if nxt not in seen_pages:
                        if (not same_host_only) or _same_host(root, nxt):
                            queue.append(nxt)

                for f_url in file_links:
                    dom_f = _domain_of(f_url)
                    if dom_f and HEALTH_ALLOWLIST and all(
                        allowed not in dom_f for allowed in HEALTH_ALLOWLIST
                    ):
                        continue

                    if files_fetched >= max_files:
                        break
                    if same_host_only and not _same_host(root, f_url):
                        continue
                    ext = Path(
                        urllib.parse.urlsplit(f_url).path
                    ).suffix.lower()
                    if ext not in ALLOWED_FILE_EXTS:
                        continue

                    fp = _download_to_tmp(f_url, tmpdir)
                    if not fp or not fp.exists():
                        continue
                    try:
                        fdocs = load_any_nonweb(str(fp))
                        for d in fdocs:
                            d.metadata = {
                                **(d.metadata or {}),
                                "url": f_url,
                                "source_page": url,
                            }
                        all_docs.extend(fdocs)
                        files_fetched += 1
                        events.append(
                            {
                                "type": "file",
                                "idx": files_fetched,
                                "url": f_url,
                                "chunks": len(fdocs),
                            }
                        )
                    except Exception:
                        pass

                time.sleep(sleep_s)
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    # Enrich with generic metadata (source_type='web')
    all_docs = enrich_docs(all_docs, source_type="web")
    return all_docs, events


# -------------------------------------------------------------------
# 5. Ingest from local upload directory
# -------------------------------------------------------------------

def ingest_files_from_dir(upload_dir: Path) -> List[Document]:
    """
    Load **all files** from a directory using the appropriate loader,
    then enrich with metadata (source_type='file').
    """
    docs: List[Document] = []
    files = [p for p in upload_dir.glob("*") if p.is_file()]
    for p in files:
        try:
            d = load_any_nonweb(str(p))
            d = enrich_docs(d, source_type="file", file_path=str(p))
            docs.extend(d)
        except Exception as e:
            print(f"Skipping {p.name}: {e}")
    return docs


# -------------------------------------------------------------------
# 6. Vector DB (FAISS + Weaviate)
# -------------------------------------------------------------------

def build_faiss_index(docs: List[Document], index_dir: Path) -> Optional[FAISS]:
    """
    Build a FAISS index from docs and save to index_dir.
    Returns the FAISS VectorStore.
    """
    if not docs:
        return None
    if index_dir.exists():
        shutil.rmtree(index_dir)
    db = FAISS.from_documents(
        docs, embedder, distance_strategy=DistanceStrategy.COSINE
    )
    db.save_local(str(index_dir))
    return db


def load_faiss_index(index_dir: Path) -> Optional[FAISS]:
    faiss_fp = index_dir / "index.faiss"
    pkl_fp = index_dir / "index.pkl"
    if not faiss_fp.exists() or not pkl_fp.exists():
        return None
    return FAISS.load_local(
        str(index_dir), embedder, allow_dangerous_deserialization=True
    )


def make_retriever(
    db,
    k: int = 3,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    filt: Optional[Dict] = None,
    use_mmr: bool = True,
):
    """
    Build a retriever from a FAISS or Weaviate vector store.
    `filt` can be used by agents (e.g., {"topic": "health"}) but
    the underlying index is common for both agents.
    """
    if isinstance(db, FAISS):
        if use_mmr:
            return db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": fetch_k,
                    "lambda_mult": lambda_mult,
                    "filter": filt or {},
                },
            )
        else:
            return db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k, "filter": filt or {}},
            )
    else:
        # Assume Weaviate VectorStore
        return db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "filter": filt or {}},
        )


# --- Weaviate helpers (if installed) ---------------------------------

def get_weaviate_client(url: str, api_key: str):
    if not weaviate or not url:
        return None
    auth = weaviate.AuthApiKey(api_key=api_key) if api_key else None
    return weaviate.Client(url=url, auth_client_secret=auth)


def _get_weav_props(client, index_name: str) -> set:
    try:
        schema = client.schema.get()
        for c in schema.get("classes", []):
            if c.get("class") == index_name:
                return {p["name"] for p in c.get("properties", [])}
    except Exception:
        pass
    return set()


def ensure_weaviate_schema(client, index_name: str, text_key: str):
    schema = client.schema.get()
    classes = {c["class"]: c for c in schema.get("classes", [])}

    base_props = [
        {"name": text_key, "dataType": ["text"]},
        {"name": "doc_id", "dataType": ["text"]},
        {"name": "chunk_id", "dataType": ["text"]},
        {"name": "source_type", "dataType": ["text"]},
        {"name": "source_url", "dataType": ["text"]},
        {"name": "file_name", "dataType": ["text"]},
        {"name": "embedding_model", "dataType": ["text"]},
        {"name": "created_at", "dataType": ["text"]},
        {"name": "source", "dataType": ["text"]},
        {"name": "url", "dataType": ["text"]},
        {"name": "source_page", "dataType": ["text"]},
        {"name": "modality", "dataType": ["text"]},
        {"name": "sheet", "dataType": ["text"]},
        {"name": "page", "dataType": ["int"]},
        {"name": "slide", "dataType": ["int"]},
        {"name": "start_sec", "dataType": ["number"]},
        {"name": "end_sec", "dataType": ["number"]},
        {"name": "source_domain", "dataType": ["text"]},
    ]

    if index_name not in classes:
        client.schema.create_class(
            {
                "class": index_name,
                "vectorizer": "none",
                "properties": base_props,
            }
        )
        return

    existing = _get_weav_props(client, index_name)
    for p in base_props:
        if p["name"] not in existing:
            try:
                client.schema.property.create(index_name, p)
            except Exception as e:
                print(f"Weaviate: could not add property {p['name']}: {e}")


def _safe_attributes(client, index_name: str, desired: List[str]) -> List[str]:
    existing = _get_weav_props(client, index_name)
    return [a for a in desired if a in existing]


def build_weaviate_index(
    docs: List[Document],
    client,
    index_name: str,
    text_key: str,
):
    if not docs or not client:
        return None

    from langchain_community.vectorstores import Weaviate as WeaviateVS

    ensure_weaviate_schema(client, index_name, text_key)
    db = WeaviateVS.from_documents(
        documents=docs,
        embedding=embedder,
        client=client,
        index_name=index_name,
        text_key=text_key,
        by_text=False,  # use nearVector
    )

    want_attrs = [
        text_key,
        "source",
        "url",
        "modality",
        "page",
        "slide",
        "sheet",
        "start_sec",
        "end_sec",
        "source_domain",
    ]
    attrs = _safe_attributes(client, index_name, want_attrs)

    db = WeaviateVS(
        client=client,
        index_name=index_name,
        text_key=text_key,
        embedding=embedder,
        by_text=False,
        attributes=attrs,
    )
    return db


def load_weaviate_index(
    client,
    index_name: str,
    text_key: str,
):
    if not client:
        return None

    from langchain_community.vectorstores import Weaviate as WeaviateVS

    ensure_weaviate_schema(client, index_name, text_key)

    want_attrs = [
        text_key,
        "source",
        "url",
        "modality",
        "page",
        "slide",
        "sheet",
        "start_sec",
        "end_sec",
        "source_domain",
    ]
    attrs = _safe_attributes(client, index_name, want_attrs)

    try:
        return WeaviateVS(
            client=client,
            index_name=index_name,
            text_key=text_key,
            embedding=embedder,
            by_text=False,
            attributes=attrs,
        )
    except Exception:
        return None


# -------------------------------------------------------------------
# 7. Load existing docs from index (for KG or inspection)
# -------------------------------------------------------------------

def faiss_docs_from_store(db) -> List[Document]:
    """
    Extract stored Documents from a FAISS VectorStore.
    Useful when the app reloads and you want to rebuild KG or stats.
    """
    try:
        return list(getattr(db, "docstore")._dict.values())
    except Exception:
        return []


def fetch_docs_from_weaviate(
    client,
    index_name: str,
    text_key: str,
    limit: int = 500,
) -> List[Document]:
    if client is None:
        return []

    schema = client.schema.get()
    classes = {c["class"]: c for c in schema.get("classes", [])}
    props = [p["name"] for p in classes[index_name]["properties"]]

    if text_key not in props:
        props.append(text_key)

    res = client.query.get(index_name, props).with_limit(limit).do()
    items = (res or {}).get("data", {}).get("Get", {}).get(index_name, []) or []

    docs = []
    for obj in items:
        text = obj.get(text_key, "") or ""
        meta = {k: v for k, v in obj.items() if k != text_key}
        if text.strip():
            docs.append(Document(page_content=text, metadata=meta))

    return docs


# -------------------------------------------------------------------
# 8. Retrieval helpers (context & preview)
# -------------------------------------------------------------------

def _cosine(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    a /= (np.linalg.norm(a) + 1e-9)
    b /= (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def get_docs(ret, query: str):
    """
    Generic wrapper for retriever.invoke.
    """
    return ret.invoke(query)


def preview_top_k_same_retriever_cos(query: str, ret, emb, k: int):
    """
    Preview top-k docs (cosine similarity between query+chunk embeddings).
    Returns list of (Document, sim_float).
    """
    docs = get_docs(ret, query)[:k]
    q_vec = emb.embed_query(query)

    rows = []
    for d in docs:
        text = getattr(d, "page_content", str(d))
        d_vec = emb.embed_query(text)
        sim = float(_cosine(q_vec, d_vec))
        rows.append((d, sim))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def format_docs_for_context(docs: List[Document], max_chars=4000) -> str:
    """
    Turn top-k docs into a context string with pseudo-citations [i].
    """
    lines = []
    for i, d in enumerate(docs, 1):
        m = d.metadata or {}
        cite = []
        if "url" in m:
            cite.append(m["url"])
        if "source" in m and "url" not in m:
            cite.append(Path(m["source"]).name)
        if "page" in m:
            cite.append(f"p.{m['page']}")
        if "slide" in m:
            cite.append(f"slide {m['slide']}")
        if "start_sec" in m or "end_sec" in m:
            cite.append(f"{m.get('start_sec','?')}–{m.get('end_sec','?')}s")
        cite_str = " | ".join(cite)
        lines.append(f"[{i}] ({cite_str})\n{d.page_content}")
    text = "\n\n".join(lines)
    return text[:max_chars]


def build_context(ret, question: str, max_chars: int = 8000):
    """
    Retrieve docs and build a single context string.
    Returns (context_str, docs_list).
    """
    docs = get_docs(ret, question)
    return format_docs_for_context(docs, max_chars=max_chars), docs


# -------------------------------------------------------------------
# 9. RAG pipeline (used by both agents)
# -------------------------------------------------------------------

RAG_TEMPLATE_DRAFT_BASE = """
{system_prompt}

Your job is to answer **only** using the text inside <context>.
Do NOT add outside knowledge or assumptions.
If the context does not contain enough information, say exactly:
"I don’t know from the provided context."

When multiple pieces of information conflict, summarize each perspective briefly.

<context>
{context}
</context>

Question:
{question}

Write a short, factual draft answer using only this context.
Include small in-text citations like [1], [2] referring to chunk numbers if helpful.

Draft Answer:
""".strip()

POLISH_TEMPLATE = """
You are editing the DRAFT_ANSWER for clarity and flow.

Rules:
- Use ONLY the facts already present in DRAFT_ANSWER.
- Do NOT introduce any new facts, numbers, or claims that are not in DRAFT_ANSWER.
- Preserve any citations like [1], [2] that appear in the draft when appropriate.
- Return ONLY the improved answer text. Do not include headings, explanations,
  or the words "DRAFT", "Paraphrased Answer", or any meta-commentary.

DRAFT_ANSWER:
{draft}
""".strip()


def run_rag_pipeline(
    question: str,
    retriever,
    llm_engine: str,
    system_prompt: str,
    openai_client=None,
    claude_client=None,
    flan_model_name: str = DEFAULT_FLAN_MODEL_NAME,
    polish: bool = False,
    max_context_chars: int = 8000,
):
    """
    Core RAG pipeline (shared by both agents):

      1) Use the provided retriever to get docs
      2) Build context string
      3) Draft answer using:
           - OpenAI (gpt-4o-mini) OR
           - Claude (haiku/sonnet) OR
           - Local FLAN-T5
      4) Optional: polish pass (no new facts)

    Returns dict:
      {
        "context":   context_str,
        "docs":      docs,
        "draft":     draft_answer,
        "answer":    final_answer,
      }
    """
    # 1) context
    context, docs = build_context(retriever, question, max_chars=max_context_chars)

    # 2) draft prompt
    rag_prompt = RAG_TEMPLATE_DRAFT_BASE.format(
        system_prompt=system_prompt, context=context, question=question
    )

    # 3) generate draft
    if llm_engine == "openai":
        if openai_client is None:
            raise ValueError("openai_client is None but llm_engine == 'openai'")
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict retrieval-only RAG assistant.",
                },
                {"role": "user", "content": rag_prompt},
            ],
        )
        draft_answer = resp.choices[0].message.content

    elif llm_engine == "claude":
        if claude_client is None:
            raise ValueError("claude_client is None but llm_engine == 'claude'")
        resp = claude_client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=1024,
            messages=[{"role": "user", "content": rag_prompt}],
        )
        draft_answer = resp.content[0].text

    else:
        # Local FLAN
        _, flan_llm = get_tokenizer_and_llm(flan_model_name)
        prompt_draft = PromptTemplate(
            template=RAG_TEMPLATE_DRAFT_BASE,
            input_variables=["system_prompt", "context", "question"],
        )
        draft_answer = (
            prompt_draft | flan_llm | StrOutputParser()
        ).invoke(
            {
                "system_prompt": system_prompt,
                "context": context,
                "question": question,
            }
        )

    final_answer = draft_answer

    # 4) optional polish
    if polish:
        if llm_engine == "openai":
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Rewrite only for clarity."},
                    {
                        "role": "user",
                        "content": POLISH_TEMPLATE.format(draft=draft_answer),
                    },
                ],
            )
            final_answer = resp.choices[0].message.content

        elif llm_engine == "claude":
            resp = claude_client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": POLISH_TEMPLATE.format(draft=draft_answer),
                    }
                ],
            )
            final_answer = resp.content[0].text

        else:
            _, flan_llm = get_tokenizer_and_llm(flan_model_name)
            prompt_polish = PromptTemplate(
                template=POLISH_TEMPLATE,
                input_variables=["draft"],
            )
            final_answer = (
                prompt_polish | flan_llm | StrOutputParser()
            ).invoke({"draft": draft_answer})

    return {
        "context": context,
        "docs": docs,
        "draft": draft_answer,
        "answer": final_answer,
    }
