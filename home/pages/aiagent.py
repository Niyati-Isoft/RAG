# app.py
# Streamlit — Multimedia Token-based RAG (Colab → Streamlit)
# -----------------------------------------------------------
# Features:
# • Upload files (PDF/DOCX/PPTX/CSV/XLSX/TXT/MD/HTML/Images/Audio/Video/Code)
# • Paste website URLs (lightweight crawler + linked-file ingestion)
# • Token-based chunking using FLAN-T5 tokenizer (HF Transformers)
# • Embeddings: sentence-transformers/all-MiniLM-L6-v2 → FAISS (L2)
# • Retriever: MMR (k, fetch_k, lambda_mult) + optional modality filters
# • RAG chain with FLAN-T5 via transformers.pipeline
# • “Preview Top-k” with  distances and metadata (page/slide/timestamps)
# • Save/Load FAISS index folder

import os, re, io, json, math, time, tempfile, shutil, urllib.parse
from pathlib import Path
from typing import List, Dict, Iterable, Optional, Tuple, Set
from collections import Counter
os.environ["TRANSFORMERS_NO_TF"] = "1"  # force Transformers to ignore TensorFlow

import streamlit as st
import pandas as pd
from PIL import Image
import weaviate
from langchain_community.vectorstores import Weaviate as WeaviateVS
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
# --- Router Agent (HEALTH vs GENERAL) ---
from routerAgent import build_local_classifier, classify_question

@st.cache_resource(show_spinner=False)
def get_router():
    return build_local_classifier("google/flan-t5-base")  # local, free

router = get_router()

# ---- Optional deps that may not exist everywhere ----
def _optional_import(name, alias=None):
    try:
        mod = __import__(name)
        return mod if alias is None else __import__(name, fromlist=[alias])
    except Exception:
        return None

pdfplumber    = _optional_import("pdfplumber")
pypdf         = _optional_import("pypdf")
docx          = _optional_import("docx")
pptx          = _optional_import("pptx")
pytesseract   = _optional_import("pytesseract")
bs4           = _optional_import("bs4", "BeautifulSoup")
trafilatura   = _optional_import("trafilatura")
whisper       = _optional_import("whisper")
moviepy       = _optional_import("moviepy")

# LangChain / HF
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter  # (not used directly but kept for parity)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import numpy as np
import requests

# ========================= Streamlit UI =========================
st.set_page_config(page_title="Multimedia Token-based RAG", layout="wide")
st.title("🎛️ Multimedia Token-based RAG")

with st.sidebar:
    st.markdown("**Vector DB Backend**")
    BACKEND = st.selectbox("Choose vector store", ["FAISS (local folder)", "Weaviate (remote)"], index=0)

    # Weaviate settings
    _WEAV = st.secrets.get("weaviate", {}) if hasattr(st, "secrets") else {}
    if BACKEND.endswith("(remote)"):
        WEAVIATE_URL    = st.text_input("Weaviate REST URL", value=_WEAV.get("url",""))
        WEAVIATE_APIKEY = st.text_input("Weaviate API Key", type="password", value=_WEAV.get("api_key",""))
        WEAV_INDEX      = st.text_input("Class / Index name", value=_WEAV.get("class","RagChunks"))
        WEAV_TEXT_KEY   = st.text_input("Text key (chunk text property)", value=_WEAV.get("text_key","text"))
    else:
        st.text_input("Index directory", key="faiss_dir", value=str(Path("./faiss_multimedia_index").resolve()))


    st.subheader("⚙️ Setup")
    UPLOAD_DIR = Path(st.text_input("Upload directory", value="./uploads"))
    INDEX_DIR  = Path(st.text_input("Index directory",  value="./faiss_multimedia_index"))
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    st.markdown("**Models**")
    HF_LLM_NAME = st.selectbox("FLAN-T5 size", ["google/flan-t5-base", "google/flan-t5-large"], index=0)
    EMBED_MODEL = st.text_input("Embeddings model", "sentence-transformers/all-MiniLM-L6-v2")

    st.markdown("**Chunking (tokens)**")
    CHUNK_SIZE_TOKENS   = st.number_input("Chunk size (tokens)", 50, 1000, 200, 10)
    CHUNK_OVERLAP_TOKENS= st.number_input("Overlap (tokens)",     0,  800,  40, 10)

    st.markdown("**Retriever (MMR)**")
    k          = st.slider("k", 1, 10, 3)
    fetch_k    = st.slider("fetch_k", 5, 50, 20, 1)
    lambda_mult= st.slider("lambda_mult", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")
    st.caption("If OCR/A/V isn’t working, install: tesseract-ocr & ffmpeg on your machine.")



METADATA_KEYS = [
    "doc_id", "chunk_id",
    "source_type", "source_url", "file_name",
    "embedding_model", "created_at",
    # keep any you already use in the UI:
    "source", "url", "modality", "page", "slide", "sheet",
    "start_sec", "end_sec",# add at the end of the list
"topic",          # ← new
"source_domain" # ← new
]



# ========================= Caches / Singletons =========================



@st.cache_resource(show_spinner=False)
def get_tokenizer_and_llm(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # NOTE: use torch_dtype (not dtype); do NOT set low_cpu_mem_usage or device_map
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
        temperature=0.6,
        repetition_penalty=1.05,
        device=0 if torch.cuda.is_available() else -1,
    )
    return tok, HuggingFacePipeline(pipeline=gen_pipe)



@st.cache_resource(show_spinner=False)
def get_embedder(name: str):
    return HuggingFaceEmbeddings(model_name=name)

tokenizer, llm = get_tokenizer_and_llm(HF_LLM_NAME)
from langchain_community.embeddings import HuggingFaceEmbeddings
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)


@st.cache_resource(show_spinner=False)
def get_weaviate_client(url: str, api_key: str):
    if not url:
        return None
    auth = weaviate.AuthApiKey(api_key=api_key) if api_key else None
    return weaviate.Client(url=url, auth_client_secret=auth)  # v3 REST client


import hashlib

def _stable_id(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()

import tldextract

HEALTH_ALLOWLIST = {
    "who.int","nih.gov","ncbi.nlm.nih.gov","pubmed.ncbi.nlm.nih.gov",
    "cdc.gov","nhs.uk","health.gov.au","eatforhealth.gov.au",
    "betterhealth.vic.gov.au","heartfoundation.org.au",
    "diabetesaustralia.com.au","cancer.org.au","dietitiansaustralia.org.au"
}

_HEALTH_KWS = {
    "nutrition","nutrient","diet","dietary","calorie","protein","carbohydrate","fat",
    "vitamin","mineral","fiber","glycemic","cholesterol","sodium","hypertension",
    "diabetes","obesity","bmi","heart disease","lipids","ldl","hdl","triglyceride",
    "pregnancy nutrition","lactation","allergy","intolerance","gluten","lactose",
    "supplement","micronutrient","macronutrient","meal plan","portions","rdis","rdi"
}

def _is_health_text(s: str, threshold_hits: int = 2) -> bool:
    if not s: return False
    s_low = s.lower()
    hits = sum(1 for kw in _HEALTH_KWS if kw in s_low)
    return hits >= threshold_hits

def _domain_of(url_or_path: str) -> str | None:
    u = str(url_or_path or "")
    if "://" not in u: return None
    ext = tldextract.extract(u)
    dom = ".".join([p for p in [ext.domain, ext.suffix] if p])
    sub = ext.subdomain
    return f"{sub}.{dom}".strip(".") if sub else dom


# --- Weaviate helpers: schema + safe attributes + builders ---

def _get_weav_props(client: weaviate.Client, index_name: str) -> set[str]:
    """Return the set of property names that currently exist on the class."""
    try:
        schema = client.schema.get()
        for c in schema.get("classes", []):
            if c.get("class") == index_name:
                return {p["name"] for p in c.get("properties", [])}
    except Exception:
        pass
    return set()

def ensure_weaviate_schema(client: weaviate.Client, index_name: str, text_key: str):
    """Create class if missing; add missing props, and report what's still missing."""
    schema = client.schema.get()
    classes = {c["class"]: c for c in schema.get("classes", [])}

    base_props = [
        {"name": text_key,         "dataType": ["text"]},
        {"name": "doc_id",         "dataType": ["text"]},
        {"name": "chunk_id",       "dataType": ["text"]},
        {"name": "source_type",    "dataType": ["text"]},
        {"name": "source_url",     "dataType": ["text"]},
        {"name": "file_name",      "dataType": ["text"]},
        {"name": "embedding_model","dataType": ["text"]},
        {"name": "created_at",     "dataType": ["text"]},
        {"name": "source",         "dataType": ["text"]},
        {"name": "url",            "dataType": ["text"]},
        {"name": "source_page",    "dataType": ["text"]},
        {"name": "modality",       "dataType": ["text"]},
        {"name": "sheet",          "dataType": ["text"]},
        {"name": "page",           "dataType": ["int"]},
        {"name": "slide",          "dataType": ["int"]},
        {"name": "start_sec",      "dataType": ["number"]},
        {"name": "end_sec",        "dataType": ["number"]},
        {"name": "topic",         "dataType": ["text"]},
        {"name": "source_domain", "dataType": ["text"]},

    ]

    # Create class if missing (vectorizer none because we supply embeddings)
    if index_name not in classes:
        client.schema.create_class({
            "class": index_name,
            "vectorizer": "none",
            "properties": base_props,
        })
        return

    # Add any missing properties
    existing_props = _get_weav_props(client, index_name)
    for p in base_props:
        if p["name"] not in existing_props:
            try:
                client.schema.property.create(index_name, p)
            except Exception as e:
                # Don't crash; just warn in UI so GraphQL won't request them
                try:
                    import streamlit as st
                    st.warning(f"Weaviate: could not add property '{p['name']}' ({e}).")
                except Exception:
                    pass

def _safe_attributes(client: weaviate.Client, index_name: str, desired: list[str]) -> list[str]:
    """Return only attributes that exist in the schema (avoids GraphQL errors)."""
    existing = _get_weav_props(client, index_name)
    return [a for a in desired if a in existing]

def build_index_weaviate(docs, client, index_name, text_key, embedding):
    if not docs:
        return None
    ensure_weaviate_schema(client, index_name, text_key)

    # Upsert (writes vectors + metadata); force nearVector path with by_text=False
    db = WeaviateVS.from_documents(
        documents=docs,
        embedding=embedding,
        client=client,
        index_name=index_name,
        text_key=text_key,
        by_text=False,  # ✅ force nearVector
    )

    # Re-wrap with a safe attribute list (intersect with schema)
    want_attrs = [text_key, "source", "url", "modality", "page", "slide", "sheet",
              "start_sec", "end_sec", "topic", "source_domain"]

    attrs = _safe_attributes(client, index_name, want_attrs)

    db = WeaviateVS(
        client=client,
        index_name=index_name,
        text_key=text_key,
        embedding=embedding,
        by_text=False,           # ✅ force nearVector in reads too
        attributes=attrs,        # ✅ only ask for props that exist
    )
    return db

def load_index_weaviate(client, index_name, text_key, embedding):
    ensure_weaviate_schema(client, index_name, text_key)

    want_attrs = [text_key, "source", "url", "modality", "page", "slide", "sheet", "start_sec", "end_sec"]
    attrs = _safe_attributes(client, index_name, want_attrs)
    try:
        return WeaviateVS(
            client=client,
            index_name=index_name,
            text_key=text_key,
            embedding=embedding,
            by_text=False,     # ✅ force nearVector
            attributes=attrs,  # ✅ avoid querying missing fields
        )
    except Exception:
        return None



weav_client = get_weaviate_client(WEAVIATE_URL, WEAVIATE_APIKEY) if BACKEND.endswith("(remote)") else None

# ========================= Chunking helpers =========================
def chunk_by_hf_tokens(text: str,
                       chunk_size: int = CHUNK_SIZE_TOKENS,
                       overlap: int    = CHUNK_OVERLAP_TOKENS) -> List[str]:
    if not text or not text.strip():
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(ids), step):
        piece = ids[start:start+chunk_size]
        if not piece:
            break
        chunks.append(tokenizer.decode(piece, skip_special_tokens=True))
        if start + chunk_size >= len(ids):
            break
    return chunks

def to_docs(chunks: Iterable[str], base_meta: Dict) -> List[Document]:
    """
    base_meta can contain: source, modality, page/slide/sheet/url/start_sec/end_sec
    We add: doc_id (stable hash of source/url) and chunk_id (increment)
    """
    # derive a stable doc id from url or source
    anchor = base_meta.get("url") or base_meta.get("source") or json.dumps(base_meta, sort_keys=True)
    doc_id = _stable_id(anchor)

    docs = []
    for i, c in enumerate([x for x in chunks if x and str(x).strip()], start=1):
        meta = base_meta.copy()
        meta["doc_id"] = doc_id
        meta["chunk_id"] = f"{doc_id}:{i}"
        docs.append(Document(page_content=c.strip(), metadata=meta))
    return docs

# ========================= Loaders =========================

def load_pdf(path: str) -> List[Document]:
    docs = []
    # Try pdfplumber first, fallback to pypdf
    if pdfplumber:
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs += to_docs(chunk_by_hf_tokens(text),
                                        {"source": path, "modality": "document", "page": i})
            return docs
        except Exception:
            pass
    if pypdf:
        try:
            reader = pypdf.PdfReader(path)
            for i, pg in enumerate(reader.pages, start=1):
                text = pg.extract_text() or ""
                if text.strip():
                    docs += to_docs(chunk_by_hf_tokens(text),
                                    {"source": path, "modality": "document", "page": i})
        except Exception:
            pass
    return docs

def load_docx(path: str) -> List[Document]:
    if not docx:
        return []
    d = docx.Document(path)
    text = "\n".join(p.text for p in d.paragraphs)
    return to_docs(chunk_by_hf_tokens(text), {"source": path, "modality": "document"})

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
            docs += to_docs(chunk_by_hf_tokens(text),
                            {"source": path, "modality": "document", "slide": si})
    return docs

def load_csv(path: str) -> List[Document]:
    df = pd.read_csv(path)
    lines = [f"{i}: " + "; ".join(f"{c}={df.loc[i, c]}" for c in df.columns) for i in range(len(df))]
    text = "\n".join(lines)
    return to_docs(chunk_by_hf_tokens(text), {"source": path, "modality": "table"})

def load_xlsx(path: str) -> List[Document]:
    docs = []
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        lines = [f"{i}: " + "; ".join(f"{c}={df.loc[i, c]}" for c in df.columns) for i in range(len(df))]
        text = "\n".join(lines)
        if text.strip():
            docs += to_docs(chunk_by_hf_tokens(text),
                            {"source": path, "modality": "table", "sheet": sheet})
    return docs

def load_textlike(path: str) -> List[Document]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return to_docs(chunk_by_hf_tokens(text), {"source": path, "modality": "text"})

def load_html(path: str) -> List[Document]:
    if not bs4:
        return []
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = bs4.BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "iframe", "svg", "header", "footer", "nav"]):
        t.decompose()
    text = soup.get_text("\n")
    text = "\n".join(line.strip() for line in text.splitlines())
    text = "\n".join([ln for ln in text.split("\n") if ln])
    if not text.strip():
        return []
    return to_docs(chunk_by_hf_tokens(text),
                   {"source": path, "modality": "text", "format": "html"})

def load_image(path: str) -> List[Document]:
    if not pytesseract:
        st.warning("pytesseract not installed; skipping OCR for images.")
        return []
    img = Image.open(path)
    text = pytesseract.image_to_string(img)
    if text.strip():
        return to_docs(chunk_by_hf_tokens(text), {"source": path, "modality": "image"})
    return []

_whisper_model = None
def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        if not whisper:
            st.warning("whisper not installed; skipping audio/video transcription.")
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
            docs.append(Document(page_content=c, metadata={
                "source": path, "modality": "audio", "start_sec": start, "end_sec": end
            }))
    return docs

def load_video(path: str) -> List[Document]:
    if not moviepy:
        st.warning("moviepy not installed; skipping audio extraction from video.")
        return []
    tmp_wav = None
    try:
        clip = moviepy.editor.VideoFileClip(path)
        tmp_wav = str(Path(tempfile.gettempdir()) / f"tmp_{int(time.time())}.wav")
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

SUPPORT_NONWEB = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".pptx": load_pptx,
    ".csv": load_csv,
    ".xlsx": load_xlsx,
    ".txt": load_textlike,
    ".md": load_textlike,
    ".html": load_html,
    ".htm": load_html,
    ".jpg": load_image, ".jpeg": load_image, ".png": load_image, ".webp": load_image,
    ".mp3": load_audio, ".wav": load_audio, ".m4a": load_audio,
    ".mp4": load_video, ".mkv": load_video, ".mov": load_video,
    ".py": load_textlike, ".js": load_textlike, ".java": load_textlike, ".cpp": load_textlike
}

def load_any_nonweb(path: str) -> List[Document]:
    ext = Path(path).suffix.lower()
    fn = SUPPORT_NONWEB.get(ext)
    if not fn:
        raise ValueError(f"Unsupported file type: {ext}")
    return fn(path)

# ========================= Metadata Enrichment =========================
def enrich_docs(docs, source_type, source_url=None, file_path=None):
    import uuid, time
    from pathlib import Path
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    doc_id = str(uuid.uuid4())
    file_name = Path(file_path).name if file_path else None
    source_domain = _domain_of(source_url or (docs[0].metadata.get("url") if docs else None))

    sample_text = (docs[0].page_content if docs else "")[:2000]
    doc_is_health = _is_health_text(sample_text)

    for i, d in enumerate(docs, 1):
        m = dict(d.metadata or {})
        m["doc_id"] = doc_id
        m["chunk_id"] = f"{doc_id}-{i:04d}"
        m["source_type"] = source_type
        if source_url:  m["source_url"] = source_url
        if file_name:   m["file_name"] = file_name
        m["embedding_model"] = EMBED_MODEL
        m["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if source_domain: m["source_domain"] = source_domain
        if doc_is_health: m["topic"] = "health"
        d.metadata = m
    return docs




# ========================= Simple website crawler (+ linked files) =========================
REQUEST_HEADERS = {"User-Agent": "RAG-Streamlit/1.0"}
REQUEST_TIMEOUT = 20
CRAWL_SLEEP_S   = 0.25
MAX_PAGES       = 20
MAX_FILES       = 10
SAME_HOST_ONLY  = True
ALLOWED_FILE_EXTS = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".md"}

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
    return urllib.parse.urlsplit(u1).netloc.lower() == urllib.parse.urlsplit(u2).netloc.lower()

def _is_text_html(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    return "text/html" in ctype or "application/xhtml" in ctype

def _visible_text_from_html(html: str) -> str:
    if not bs4:
        return ""
    soup = bs4.BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "iframe"]):
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
        u = _normalize_url(_abs_url(base_url, href.split("#", 1)[0]))
        ext = Path(urllib.parse.urlsplit(u).path).suffix.lower()
        if ext in ALLOWED_FILE_EXTS:
            file_links.add(u)
        else:
            page_links.add(u)
    return page_links, file_links

def _download_to_tmp(url: str, tmpdir: Path) -> Optional[Path]:
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS, stream=True)
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
    return [Document(page_content=c, metadata={"url": url, "source": url, "modality": "web"})
            for c in chunk_by_hf_tokens(text)]

def load_website_with_files(roots: List[str],
                            same_host_only: bool = SAME_HOST_ONLY,
                            max_pages: int = MAX_PAGES,
                            max_files: int = MAX_FILES,
                            sleep_s: float = CRAWL_SLEEP_S,
                            return_log: bool = False):
    """
    Crawl pages + linked files. If return_log=True, returns (docs, log)
    where log is a list of dicts describing events for pretty display.
    """
    all_docs: List[Document] = []
    events = []  # for pretty log
    tmpdir = Path(tempfile.mkdtemp(prefix="rag_web_"))
    try:
        for root in roots:
            if not root:
                continue
            root = _normalize_url(root.strip())
            if return_log:
                events.append({"type": "start", "url": root})

            seen_pages, queue = set(), [root]
            pages_fetched, files_fetched = 0, 0

            while queue and pages_fetched < max_pages:
                url = queue.pop(0)

                # Health-domain gate (optional allowlist)
                dom = _domain_of(url)
                if dom and HEALTH_ALLOWLIST and all(allowed not in dom for allowed in HEALTH_ALLOWLIST):
                    continue

                if url in seen_pages:
                    continue
                seen_pages.add(url)

                if same_host_only and not _same_host(root, url):
                    continue

                try:
                    resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=REQUEST_HEADERS)
                except Exception:
                    continue
                if resp.status_code != 200 or not _is_text_html(resp):
                    continue

                # Page text → chunks
                text = _extract_text(url, resp.text)
                page_docs = _token_chunk_to_docs(text, url)
                all_docs.extend(page_docs)
                pages_fetched += 1

                if return_log:
                    events.append({
                        "type": "page",
                        "idx": pages_fetched,
                        "url": url,
                        "chunks": len(page_docs)
                    })

                # Links
                page_links, file_links = _gather_links(url, resp.text)
                for nxt in page_links:
                    if nxt not in seen_pages:
                        if (not same_host_only) or _same_host(root, nxt):
                            queue.append(nxt)

                # Files on the page
                for f_url in file_links:
                    # Health-domain gate for file links
                    dom_f = _domain_of(f_url)
                    if dom_f and HEALTH_ALLOWLIST and all(allowed not in dom_f for allowed in HEALTH_ALLOWLIST):
                        continue

                    if files_fetched >= max_files:
                        break
                    if same_host_only and not _same_host(root, f_url):
                        continue
                    ext = Path(urllib.parse.urlsplit(f_url).path).suffix.lower()
                    if ext not in ALLOWED_FILE_EXTS:
                        continue

                    fp = _download_to_tmp(f_url, tmpdir)
                    if not fp or not fp.exists():
                        continue
                    try:
                        fdocs = load_any_nonweb(str(fp))
                        for d in fdocs:
                            d.metadata = {**d.metadata, "url": f_url, "source_page": url}
                        all_docs.extend(fdocs)
                        files_fetched += 1
                        if return_log:
                            events.append({
                                "type": "file",
                                "idx": files_fetched,
                                "url": f_url,
                                "chunks": len(fdocs)
                            })
                    except Exception:
                        pass

                time.sleep(sleep_s)
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    if return_log:
        return all_docs, events
    return all_docs



# ---- Pretty chunking preview (works for web + local files) ----

from urllib.parse import urlparse
from collections import defaultdict


FILE_EXTS = {".pdf", ".docx", ".pptx", ".csv", ".xlsx", ".txt", ".md", ".html"}

def _ext_from_path(path: str) -> str:
    try:
        # works for both URLs and local paths
        p = urlparse(path).path if "://" in path else path
        ext = os.path.splitext(p)[1].lower()
        return ext
    except Exception:
        return ""

def _is_page_like(url: str) -> bool:
    """
    Treat as a 'page' if it ends with / or has no file ext (typical website pages).
    """
    p = urlparse(url).path
    ext = os.path.splitext(p)[1].lower()
    return (not ext) or ext in {"", ".html", ".htm", ".php", ".asp", ".aspx", ".jsp"}

def show_chunking_preview(docs, title="Ingestion (token-chunk → Documents)", max_items=40):
    """
    docs: list[langchain_core.documents.Document]
    Renders a summary like:
      Start: https://example.com
        ✓ Page [1]: https://example.com/a  (+6 chunks)
        ↳ File [1]: https://example.com/report.pdf  (+112 chunks)
      File: /path/to/local.pdf  (+37 chunks)
    """
    if not docs:
        st.info("No documents to preview.")
        return

    # Split by modality for friendlier output
    by_modality = defaultdict(list)
    for d in docs:
        m = (d.metadata or {}).get("modality") or "unknown"
        by_modality[m].append(d)

    st.markdown(f"### 2) 🔁 {title}")

    # ---------- WEB ----------
    web_docs = by_modality.get("web", [])
    if web_docs:
        # Determine a "root" (first host)
        first_url = (web_docs[0].metadata.get("url")
                     or web_docs[0].metadata.get("source") or "")
        parsed = urlparse(first_url) if "://" in first_url else None
        root = f"{parsed.scheme}://{parsed.netloc}" if parsed else first_url
        st.markdown(f"🌐 **Start:** [{root}]({root})")

        page_counts = defaultdict(int)
        file_counts = defaultdict(int)
        for d in web_docs:
            url = d.metadata.get("url") or d.metadata.get("source") or ""
            if not url:
                continue
            if _is_page_like(url):
                page_counts[url] += 1
            else:
                file_counts[url] += 1

        # Pages
        if page_counts:
            for i, (url, n) in enumerate(sorted(page_counts.items(), key=lambda x: x[1], reverse=True)[:max_items], 1):
                st.markdown(f"  ✓ **Page [{i}]**: [{url}]({url})  _( +{n} chunks )_")

        # Files (PDF/DOCX/…)
        if file_counts:
            for i, (url, n) in enumerate(sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:max_items], 1):
                ext = _ext_from_path(url) or ""
                st.markdown(f"  ↳ **File [{i}]**: [{url}]({url})  _( +{n} chunks )_")

    # ---------- LOCAL / OTHER (documents, tables, images, audio...) ----------
    other_modalities = [m for m in by_modality.keys() if m != "web"]
    if other_modalities:
        st.markdown("---")
    for mod in other_modalities:
        docs_m = by_modality[mod]
        # Group by a stable display key (prefer file_name, then source)
        groups = defaultdict(int)
        for d in docs_m:
            meta = d.metadata or {}
            key = (meta.get("file_name")
                   or meta.get("source")
                   or meta.get("doc_id")
                   or "Unknown")
            groups[key] += 1

        icon = {
            "document": "📄",
            "table": "📊",
            "image": "🖼️",
            "audio": "🎧",
            "video": "🎬",
            "text": "📝",
            "unknown": "📦"
        }.get(mod, "📦")

        st.markdown(f"{icon} **{mod.title()} sources**")
        for i, (key, n) in enumerate(sorted(groups.items(), key=lambda x: x[1], reverse=True)[:max_items], 1):
            # Make local path look nice; link only if it’s a URL
            if "://" in str(key):
                st.markdown(f"  • **{i}.** [{key}]({key})  _( +{n} chunks )_")
            else:
                ext = _ext_from_path(str(key))
                label = f"{os.path.basename(str(key))}" if os.path.exists(str(key)) else str(key)
                if ext:
                    st.markdown(f"  • **{i}.** {label}  `{ext}`  _( +{n} chunks )_")
                else:
                    st.markdown(f"  • **{i}.** {label}  _( +{n} chunks )_")


# ========================= Build / Load FAISS =========================


def build_index(docs: List[Document], index_dir: Path) -> Optional[FAISS]:
    if not docs:
        return None
    if index_dir.exists():
        shutil.rmtree(index_dir)
    db = FAISS.from_documents(
        docs,
        embedder,
        distance_strategy=DistanceStrategy.COSINE
    )
    db.save_local(str(index_dir))
    return db

def load_index(index_dir: Path):
    faiss_fp = index_dir / "index.faiss"
    pkl_fp   = index_dir / "index.pkl"
    if not faiss_fp.exists() or not pkl_fp.exists():
        st.info(f"No index found yet in {index_dir}. Please ingest data and click **Build index from current docs**.")
        return None
    return FAISS.load_local(str(index_dir), embedder, allow_dangerous_deserialization=True)


def make_retriever(db: FAISS, k=3, fetch_k=20, lambda_mult=0.5, filt: Dict=None):
    return db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult, "filter": (filt or {})}
    )

# ========================= Preview Top-k (with L2) =========================
#def _l2(a, b):
    #a = np.array(a, dtype=np.float32)
    #b = np.array(b, dtype=np.float32)
    #return float(np.linalg.norm(a - b))



# ---------- Utilities ----------
import numpy as np
from functools import lru_cache

def _cosine(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    a /= (np.linalg.norm(a) + 1e-9)
    b /= (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))




# Cache embeddings to avoid re-encoding the same text in preview
@lru_cache(maxsize=10_000)
def _embed_cached(emb, text: str):
    return tuple(emb.embed_query(text))  # tuple so it becomes hashable for cache


def preview_top_k_same_retriever_cos(query: str, ret, emb, k: int):
    """Preview top-k docs using cosine similarity without any external helpers."""
    docs = get_docs(ret, query)[:k]
    q_vec = emb.embed_query(query)

    rows = []
    for d in docs:
        # Some retrievers return Document; fall back to string if needed
        text = getattr(d, "page_content", str(d))
        d_vec = emb.embed_query(text)
        sim = _cosine(q_vec, d_vec)
        rows.append((d, sim))

    # Higher cosine = better match
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def format_docs_for_context(docs: List[Document], max_chars=4000) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        cite = []
        if "url" in m: cite.append(m["url"])
        if "source" in m and "url" not in m: cite.append(Path(m["source"]).name)
        if "page" in m: cite.append(f"p.{m['page']}")
        if "slide" in m: cite.append(f"slide {m['slide']}")
        if "start_sec" in m or "end_sec" in m:
            cite.append(f"{m.get('start_sec','?')}–{m.get('end_sec','?')}s")
        cite_str = " | ".join(cite)
        lines.append(f"[{i}] ({cite_str})\n{d.page_content}")
    text = "\n\n".join(lines)
    return text[:max_chars]



# ========================= Session State =========================
if "all_docs_nonweb" not in st.session_state: st.session_state.all_docs_nonweb = []
if "web_docs"        not in st.session_state: st.session_state.web_docs        = []
if "db"              not in st.session_state: st.session_state.db              = load_index(INDEX_DIR)
if "retriever"       not in st.session_state and st.session_state.db:
    st.session_state.retriever = make_retriever(st.session_state.db, k, fetch_k, lambda_mult)
retriever = st.session_state.get("retriever", None)

# ========================= Uploader & URL Box =========================
st.header("1) 📁 Upload files & 🌐 Add URLs")

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader(
        "Upload multiple files",
        type=list({ext.strip(".") for ext in SUPPORT_NONWEB.keys()}),
        accept_multiple_files=True
    )
    if uploaded:
        saved = []
        for up in uploaded:
            out = UPLOAD_DIR / up.name
            # ensure unique name
            base, ext = out.stem, out.suffix
            i = 1
            while out.exists():
                out = UPLOAD_DIR / f"{base} ({i}){ext}"; i += 1
            with open(out, "wb") as f:
                f.write(up.read())
            saved.append(str(out))
        st.success(f"Saved {len(saved)} file(s) to {UPLOAD_DIR}")

with col2:
    urls_text = st.text_area("Paste website URLs (comma or newline separated)")
    same_host = st.checkbox("Stay on same host", True)
    max_pages = st.slider("Max pages/site", 5, 100, MAX_PAGES, 5)
    max_files = st.slider("Max linked files/site", 0, 50, MAX_FILES, 1)

# Show file stats
existing_files = [str(p) for p in UPLOAD_DIR.glob("*") if p.is_file()]
if existing_files:
    df_stats = []
    for p in existing_files:
        try:
            stt = Path(p).stat()
            df_stats.append({"file": Path(p).name, "ext": Path(p).suffix.lower(),
                             "size_kb": round(stt.st_size/1024,1)})
        except Exception:
            df_stats.append({"file": Path(p).name, "ext": Path(p).suffix.lower(),
                             "size_kb": None})
    st.dataframe(pd.DataFrame(df_stats), use_container_width=True)

# ========================= Ingestion Actions =========================

st.header("2) 🔄 Ingestion (token-chunk → Documents)")
def render_crawl_log(events: List[dict]):
    if not events:
        return
    lines = []
    for e in events:
        if e["type"] == "start":
            lines.append(f"🌐 **Start:** {e['url']}")
        elif e["type"] == "page":
            lines.append(f"✓ **Page [{e['idx']}]:** {e['url']}  *(+{e['chunks']} chunks)*")
        elif e["type"] == "file":
            lines.append(f"↳ **File [{e['idx']}]:** {e['url']}  *(+{e['chunks']} chunks)*")
    st.markdown("\n\n".join(lines))

c1, c2 = st.columns(2)
with c1:
    if st.button("Ingest uploaded files"):
            all_docs_nonweb = []
            for p in existing_files:
                try:
                    docs = load_any_nonweb(p)
                    docs = enrich_docs(docs, source_type="file", file_path=p)  # ← add this
                    all_docs_nonweb.extend(docs)
                except Exception as e:
                    st.warning(f"Skip {Path(p).name}: {e}")
            st.session_state.all_docs_nonweb = all_docs_nonweb
            st.success(f"Loaded {len(all_docs_nonweb)} chunks from {len(existing_files)} files.")
            show_chunking_preview(all_docs_nonweb, title="Files (token-chunks → Documents)")  # optional preview


with c2:
    if st.button("Crawl & ingest websites"):
        raw = urls_text.replace("\n", ",")
        web_urls = [u.strip() for u in raw.split(",") if u.strip()]
        if not web_urls:
            st.warning("No URLs provided.")
        else:
            web_docs, crawl_log = load_website_with_files(
                web_urls,
                same_host_only=same_host,
                max_pages=max_pages,
                max_files=max_files,
                return_log=True
            )
            # Enrich: if you want the root URL stored explicitly, do it per-doc from existing metadata
            # (keeps correct source_url even when multiple roots are crawled)
            for d in web_docs:
                src = (d.metadata or {}).get("url") or (d.metadata or {}).get("source")
            web_docs = enrich_docs(web_docs, source_type="web")  # ← add this (source_url taken from doc meta)

            st.session_state.web_docs = web_docs
            st.success(f"Loaded {len(web_docs)} web chunks.")
            with st.expander("Crawl log", expanded=True):
                render_crawl_log(crawl_log)
            show_chunking_preview(web_docs, title="Web (token-chunks → Documents)")  # optional preview



st.caption(f"Non-web chunks: {len(st.session_state.all_docs_nonweb)}  |  Web chunks: {len(st.session_state.web_docs)}")

# ========================= Build / Load Index =========================
st.header("3) 🗂️ Vector DB")
colb1, colb2, colb3 = st.columns(3)

with colb1:
    if st.button("Build index from current docs"):
        all_docs = st.session_state.all_docs_nonweb + st.session_state.web_docs
        if BACKEND.startswith("FAISS"):
            db = build_index(all_docs, INDEX_DIR)
            st.session_state.db = db
            st.session_state.retriever = make_retriever(db, k, fetch_k, lambda_mult) if db else None
            st.success(f"[FAISS] Index built at {INDEX_DIR}. Docs: {len(all_docs)}")
        else:
            if not weav_client:
                st.error("Weaviate not configured.")
            else:
                db = build_index_weaviate(all_docs, weav_client, WEAV_INDEX, WEAV_TEXT_KEY, embedder)
                st.session_state.db = db
                # 🔁 Use plain similarity for Weaviate (avoid MMR GraphQL path)
                st.session_state.retriever = db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k}
                ) if db else None
                st.success(f"[Weaviate] Upserted {len(all_docs)} chunks into '{WEAV_INDEX}'.")


with colb2:
    if st.button("Load existing index"):
        if BACKEND.startswith("FAISS"):
            db = load_index(INDEX_DIR)
            st.session_state.db = db
            st.session_state.retriever = make_retriever(db, k, fetch_k, lambda_mult) if db else None
            st.success("Index loaded." if db else "No index found.")
        else:
            if not weav_client:
                st.error("Weaviate not configured.")
            else:
                db = load_index_weaviate(weav_client, WEAV_INDEX, WEAV_TEXT_KEY, embedder)
                st.session_state.db = db
                # 🔁 Use plain similarity for Weaviate
                st.session_state.retriever = db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k}
                ) if db else None
                st.success(f"[Weaviate] Connected to '{WEAV_INDEX}'.")

with colb3:
    if st.button("Clear index folder"):
        if BACKEND.startswith("FAISS"):
            try:
                shutil.rmtree(INDEX_DIR, ignore_errors=True)
                INDEX_DIR.mkdir(parents=True, exist_ok=True)
                st.session_state.db = None
                st.session_state.retriever = None
                st.success("FAISS index folder cleared.")
            except Exception as e:
                st.error(f"Failed to clear: {e}")
        else:
            st.info("For Weaviate, delete objects via Weaviate console/admin script (not from the app).")

# ========================= Query / Preview / Answer =========================
st.header("4) 🔎 Retrieve & 💬 Ask")
q = st.text_input("Your question", value="High protein meal ideas")
# ---- Route the query with the router agent ----
label, debug_info = classify_question(q or "", clf_pipe=router)
st.caption(f"🧭 Router decision: **{label.upper()}**")
with st.expander("Router debug", expanded=False):
    st.json(debug_info)

# Optional manual override
manual_override = st.checkbox("Override router → Force HEALTH mode", value=False)
effective_label = "health" if manual_override else label



# ========================= Question Token Summary =========================
st.subheader("🔢 Question Token Summary")

count_mode = st.radio(
    "Token counter",
    ["Embedder (MiniLM)", "LLM (FLAN-T5)"],
    horizontal=True,
    index=0,
)

from transformers import AutoTokenizer as HFAutoTokenizer

@st.cache_resource(show_spinner=False)
def _mini_tokenizer():
    return HFAutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Pick ONE tokenizer for both question and preview chunks
tok_for_counts = _mini_tokenizer() if count_mode.startswith("Embedder") else tokenizer

if q.strip():
    q_tokens = len(tok_for_counts.encode(q, add_special_tokens=False))
    st.caption(f"Question tokens ({count_mode}): **{q_tokens}**")
else:
    st.caption("Enter a question above to see token counts.")

polish = st.checkbox("Polish the final answer")

# ---------- helpers ----------
def get_docs(ret, query: str):
    return ret.invoke(query)

def build_context(ret, question: str, max_chars: int = 8000):
    docs = get_docs(ret, question)
    return format_docs_for_context(docs, max_chars=max_chars), docs



# ---------- Run retrieval only if retriever and question exist ----------
if retriever and q:
    from langchain_community.vectorstores import Weaviate as WeaviateVS

    # Filter only if HEALTH; GENERAL uses all docs
    health_filter = {"topic": "health"} if effective_label == "health" else {}

    db = st.session_state.db
    if isinstance(db, FAISS):
        ret = make_retriever(db, k, fetch_k, lambda_mult, filt=health_filter)
    else:
        ret = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "filter": health_filter}
        )

        # ---- Weaviate nearVector + safe attrs hotfix ----
        def _force_near_vector_and_safe_attrs(ret_):
            vs = getattr(ret_, "vectorstore", None)
            if not isinstance(vs, WeaviateVS):
                return
            try: vs._by_text = False
            except: pass
            try:
                schema = vs._client.schema.get() if hasattr(vs, "_client") else vs.client.schema.get()
                cls = next((c for c in schema.get("classes", [])
                            if c.get("class") in [getattr(vs, "_index_name", None), getattr(vs, "index_name", None)]), None)
                existing = {p["name"] for p in (cls.get("properties", []) if cls else [])}
                attrs = getattr(vs, "_attributes", None) or getattr(vs, "attributes", None)
                text_key = getattr(vs, "_text_key", None) or getattr(vs, "text_key", None)
                safe = [a for a in (attrs or []) if (a in existing) or (a == text_key)] or [text_key]
                if hasattr(vs, "_attributes"): vs._attributes = safe
                else: vs.attributes = safe
            except:
                text_key = getattr(vs, "_text_key", None) or getattr(vs, "text_key", None)
                if text_key:
                    if hasattr(vs, "_attributes"): vs._attributes = [text_key]
                    else: vs.attributes = [text_key]

        _force_near_vector_and_safe_attrs(ret)


    # ---------- Stage 0: show preview ----------
    st.subheader("Preview Top-k")
    rows = preview_top_k_same_retriever_cos(q, ret, embedder, k)

    from transformers import AutoTokenizer
    @st.cache_resource(show_spinner=False)
    def _mini_tokenizer():
        return AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    tok = _mini_tokenizer()

    for i, (doc, sim) in enumerate(rows, 1):
        meta = dict(getattr(doc, "metadata", {}) or {})
        text = getattr(doc, "page_content", "")

        cite = []
        if "url" in meta: cite.append(meta["url"])
        if "source" in meta and "url" not in meta:
            from pathlib import Path
            cite.append(Path(meta["source"]).name)
        if "page" in meta:  cite.append(f"p.{meta['page']}")
        if "slide" in meta: cite.append(f"slide {meta['slide']}")
        if "start_sec" in meta or "end_sec" in meta:
            cite.append(f"{meta.get('start_sec','?')}–{meta.get('end_sec','?')}s")

        st.markdown(f"**[{i}] Cosine = {sim:.4f}**")
        if cite:
            st.caption(" • ".join(cite))

        st.caption(text[:400] + (" ..." if len(text) > 400 else ""))
        token_len = len(tok_for_counts.encode(text, add_special_tokens=False))
        st.caption(f"Tokens ({count_mode}): {token_len}")
        with st.expander("🔍 Chunk metadata", expanded=False):
            st.json(meta)
        st.markdown("---")




        # ---------- Stage 1: RAG draft (factual) ----------
    RAG_TEMPLATE_DRAFT_HEALTH = RAG_TEMPLATE_DRAFT = """
            You are a retrieval-augmented assistant limited to general **health and nutrition education**.
            Stay within this domain. Do NOT provide diagnosis, treatment, or individualized medical advice.
            If the context is insufficient or outside domain, reply exactly:
            "I don’t know from the provided context."

            <context>
            {context}
            </context>

            Question:
            {question}

            Write a short, factual educational draft answer **only** using the context.
            If appropriate, include brief citations like [1], [2] referring to chunk numbers.
            If the user describes urgent or severe symptoms, include this sentence at the end:
            "This information is educational and not a medical diagnosis. Please seek professional care."
            """.strip()
    RAG_TEMPLATE_DRAFT_GENERAL = """
    You are a precise retrieval-augmented assistant.

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
    RAG_TEMPLATE_DRAFT = RAG_TEMPLATE_DRAFT_HEALTH if effective_label == "health" else RAG_TEMPLATE_DRAFT_GENERAL
    prompt_draft = PromptTemplate(template=RAG_TEMPLATE_DRAFT, input_variables=["question", "context"])

    # Build context once and reuse
    ctx, docs = build_context(ret, q)
    draft_answer = (prompt_draft | llm | StrOutputParser()).invoke({"question": q, "context": ctx})

    # ---------- Stage 2: Polish (clarity only; no new facts) ----------
    if polish:
        POLISH_TEMPLATE = """
                You are an expert editor.
                Paraphrase the following answer to make it more natural, concise, and fluent in English.
                Do NOT add, remove, or change any facts or meanings.
                Keep the tone neutral and factual — only rephrase wording and improve sentence flow.""" + ("""
                Stay within general health & nutrition education and avoid medical advice.
            """ if effective_label == "health" else "") + """
                

                DRAFT ANSWER:
                {draft}

                Paraphrased Answer:
                """.strip()


        prompt_polish = PromptTemplate(template=POLISH_TEMPLATE, input_variables=["draft", "context"])
        polished_answer = (prompt_polish | llm | StrOutputParser()).invoke({"draft": draft_answer, "context": ctx})

        st.subheader("Polished Answer:")
        st.write(polished_answer)

    else:
        st.subheader("Final Answer:")
        st.write(draft_answer)

    st.markdown("---")
