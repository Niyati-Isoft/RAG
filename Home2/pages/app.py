# app.py
# Streamlit UI that:
#  - lets you upload files / type URLs
#  - calls rag_core helpers on button click
#  - builds shared FAISS index
#  - routes question → health or general agent (both use same retriever)

import os
from pathlib import Path
import streamlit as st

from openai import OpenAI
from anthropic import Anthropic

from rag_core import (
    ingest_files_from_dir,
    crawl_and_ingest_web,
    build_faiss_index,
    load_faiss_index,
    make_retriever,
)

from routeragentai import (
    build_local_classifier,
    classify_question_llm,
    classify_question_cloud_openai,
    classify_question_cloud_claude,
)

from agent_health import answer_health_question   # you’ll write this
from agent_general import answer_general_question # you’ll write this

# ----------------- Config -----------------
st.set_page_config(page_title="Multimedia RAG", layout="wide")
st.title("🎛️ Multimedia Token-based RAG (with Router)")

UPLOAD_DIR = Path("./uploads")
INDEX_DIR  = Path("./faiss_multimedia_index")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- Session state -----------------
if "docs" not in st.session_state:      st.session_state.docs = []
if "web_docs" not in st.session_state:  st.session_state.web_docs = []
if "db" not in st.session_state:        st.session_state.db = None
if "retriever" not in st.session_state: st.session_state.retriever = None

# ----------------- Sidebar: LLM + Router choice -----------------
with st.sidebar:
    st.markdown("### 🔑 LLM Configuration")
    OPENAI_KEY = st.secrets.get("openai", {}).get("OPENAI_API_KEY")
    CLAUDE_KEY = st.secrets.get("anthropic", {}).get("ANTHROPIC_API_KEY")

    client_openai  = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
    client_claude  = Anthropic(api_key=CLAUDE_KEY) if CLAUDE_KEY else None

    LLM_CHOICE = st.radio(
        "Answer engine",
        ["OpenAI", "Claude", "Local FLAN-T5"],
        index=0,
    )

    if LLM_CHOICE == "OpenAI" and client_openai:
        ACTIVE_LLM = "openai"
    elif LLM_CHOICE == "Claude" and client_claude:
        ACTIVE_LLM = "claude"
    else:
        ACTIVE_LLM = "flan"

    # Router engine
    ROUTER_ENGINE = st.radio(
        "Router engine (for HEALTH vs GENERAL)",
        ["OpenAI", "Claude", "Local FLAN-T5"],
        index=2,
    )

    # Build router once
    @st.cache_resource(show_spinner=False)
    def get_router(engine, client_openai, client_claude):
        if engine == "OpenAI" and client_openai:
            return "openai", client_openai
        if engine == "Claude" and client_claude:
            return "claude", client_claude
        # local
        return "flan", build_local_classifier("google/flan-t5-base")

    router_kind, router_obj = get_router(ROUTER_ENGINE, client_openai, client_claude)

# ----------------- 1) Upload files & URLs -----------------
st.header("1) 📁 Upload files & 🌐 Add URLs")

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader(
        "Upload multiple files",
        type=["pdf","docx","pptx","csv","xlsx","txt","md","html","htm",
              "jpg","jpeg","png","webp","mp3","wav","m4a","mp4","mkv","mov",
              "py","js","java","cpp"],
        accept_multiple_files=True,
    )
    if uploaded:
        saved_paths = []
        for up in uploaded:
            out = UPLOAD_DIR / up.name
            base, ext = out.stem, out.suffix
            i = 1
            while out.exists():
                out = UPLOAD_DIR / f"{base} ({i}){ext}"
                i += 1
            with open(out, "wb") as f:
                f.write(up.read())
            saved_paths.append(str(out))
        st.success(f"Saved {len(saved_paths)} file(s) to {UPLOAD_DIR}")

with col2:
    urls_text = st.text_area("Paste website URLs (comma or newline separated)")
    same_host = st.checkbox("Stay on same host", True)
    max_pages = st.slider("Max pages/site", 5, 100, 20, 5)
    max_files = st.slider("Max linked files/site", 0, 50, 10, 1)

# ----------------- 2) 🔄 Ingestion -----------------
st.header("2) 🔄 Ingestion (token-chunk → Documents)")

c1, c2 = st.columns(2)
with c1:
    if st.button("Ingest uploaded files"):
        docs = ingest_files_from_dir(UPLOAD_DIR)
        st.session_state.docs = docs
        st.success(f"Ingested {len(docs)} chunks from files.")

with c2:
    if st.button("Crawl & ingest websites"):
        raw = urls_text.replace("\n", ",")
        web_urls = [u.strip() for u in raw.split(",") if u.strip()]
        if not web_urls:
            st.warning("No URLs provided.")
        else:
            web_docs, crawl_log = crawl_and_ingest_web(
                web_urls,
                same_host_only=same_host,
                max_pages=max_pages,
                max_files=max_files,
            )
            st.session_state.web_docs = web_docs
            st.success(f"Ingested {len(web_docs)} web chunks.")
            with st.expander("Crawl log", expanded=False):
                for e in crawl_log:
                    if e["type"] == "start":
                        st.write(f"🌐 Start: {e['url']}")
                    elif e["type"] == "page":
                        st.write(f"✓ Page {e['idx']}: {e['url']} (+{e['chunks']} chunks)")
                    elif e["type"] == "file":
                        st.write(f"↳ File {e['idx']}: {e['url']} (+{e['chunks']} chunks)")

st.caption(
    f"File chunks: {len(st.session_state.docs)} | Web chunks: {len(st.session_state.web_docs)}"
)

# ----------------- 3) 🗂️ Vector DB (shared for both agents) -----------------
st.header("3) 🗂️ Vector DB (shared FAISS index)")

colb1, colb2 = st.columns(2)
with colb1:
    if st.button("Build FAISS index from current docs"):
        all_docs = st.session_state.docs + st.session_state.web_docs
        if not all_docs:
            st.warning("No docs to index – ingest files or websites first.")
        else:
            db = build_faiss_index(all_docs, INDEX_DIR)
            st.session_state.db = db
            st.session_state.retriever = make_retriever(
                db,
                k=3,
                fetch_k=20,
                lambda_mult=0.5,
                filt=None,       # ✅ same index for both agents
                use_mmr=True,
            )
            st.success(f"Index built with {len(all_docs)} documents.")

with colb2:
    if st.button("Load existing FAISS index"):
        db = load_faiss_index(INDEX_DIR)
        if db is None:
            st.warning("No FAISS index found in ./faiss_multimedia_index")
        else:
            st.session_state.db = db
            st.session_state.retriever = make_retriever(
                db,
                k=3,
                fetch_k=20,
                lambda_mult=0.5,
                filt=None,
                use_mmr=True,
            )
            st.success("Loaded FAISS index and retriever.")

# ----------------- 4) 🔎 Ask → Route → Agent -----------------
st.header("4) 🔎 Ask a question")

q = st.text_input("Your question", value="High protein meal ideas")

if q and st.session_state.retriever is None:
    st.warning("Build or load the index first.")
elif q:
    # ---- Route to HEALTH / GENERAL ----
    if router_kind == "openai":
        label, debug_info = classify_question_cloud_openai(q, router_obj)
    elif router_kind == "claude":
        label, debug_info = classify_question_cloud_claude(q, router_obj)
    else:
        label, debug_info = classify_question_llm(q, clf_pipe=router_obj)

    st.caption(f"🧭 Router decision: **{label.upper()}**")
    with st.expander("Router debug", expanded=False):
        st.json(debug_info)

    polish = st.checkbox("Polish final answer", value=True)

    # ---- Call the right agent (both use SAME retriever) ----
    if st.button("Get answer"):
        if label == "health":
            answer, dbg = answer_health_question(
                question=q,
                retriever=st.session_state.retriever,
                llm_engine=ACTIVE_LLM,
                openai_client=client_openai,
                claude_client=client_claude,
                polish=polish,
            )
        else:
            answer, dbg = answer_general_question(
                question=q,
                retriever=st.session_state.retriever,
                llm_engine=ACTIVE_LLM,
                openai_client=client_openai,
                claude_client=client_claude,
                polish=polish,
            )

        st.subheader("Answer")
        st.write(answer)
        with st.expander("Debug RAG info", expanded=False):
            st.json(dbg)
