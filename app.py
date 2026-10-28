import streamlit as st
import os
import re
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import tempfile
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

# LangChain Imports
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# Core Modular Package Imports
from core.config import (
    PAGE_TITLE,
    EMBEDDING_MODEL,
    PERSIST_DIR,
    GEMINI_VISION_MODEL,
    FactCategory,
    CONFIDENCE_WEIGHTS,
    _TABULAR_DOC_PATTERNS,
    _WEB_QUERY_PATTERNS,
    _EXPLICIT_WEB_PHRASES,
    _PERSONAL_QUERY_KEYWORDS,
    _WEB_FOLLOWUP_PATTERN,
    _WEB_FOLLOWUP_NUMBER,
    _ORDINAL_MAP,
    _BROAD_QUERY_PATTERNS,
    _RECENCY_META_PATTERNS,
    _MEMORY_RECALL_PATTERNS,
    _DATA_KEYWORDS,
    _CASUAL_PATTERNS,
    _STALE_FACT_DAYS,
    _STALE_CHECK_INTERVAL_TURNS,
    _SUMMARIZE_EVERY_N_TURNS,
    _cloud_prompt_text,
    _local_prompt_text,
    bot_logger,
)

from core.ingestion import (
    GEMINI_VISION_AVAILABLE,
    extract_text_digital,
    extract_text_gemini_vision,
    extract_text_rapidocr_fallback,
    extract_text_rapidocr,
    extract_text_from_pdf,
    smart_chunk_documents,
)

from core.web import (
    DDGS_AVAILABLE,
    URL_FETCH_AVAILABLE,
    _HTMLTextExtractor,
    _fetch_url_text,
    search_web_ddg,
    _search_web,
    _is_personal_query,
    _should_auto_web_search,
    _is_web_query,
    _is_casual_message,
    _check_web_followup,
)

from core.memory import (
    init_databases,
    get_vector_stores as core_get_vector_stores,
    _match_document_source,
    _detect_referenced_sources,
    _recency_boosted_episodic_search,
    _get_recent_episodic_memories,
    _get_all_identity_facts,
    _retrieve_full_source_chunks,
    retrieve_from_vault,
    _consolidate_episodic_memory,
    _maybe_summarize_session,
    _clear_collection,
)

from core.identity import (
    _detect_contradiction,
    _add_to_clarification_queue,
    save_extracted_facts,
    extract_personal_facts,
    extract_facts_from_document,
    _check_stale_facts,
    _build_clarification_prompt,
    _process_clarification_response,
    run_background_extraction,
)

load_dotenv()

# ╔══════════════════════════════════════════════════════╗
# ║              CONFIGURATION                           ║
# ╚══════════════════════════════════════════════════════╝
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(f"🧠 {PAGE_TITLE}")


# ╔══════════════════════════════════════════════════════╗
# ║  1. TRIPLE-MEMORY ARCHITECTURE (ChromaDB)            ║
# ╚══════════════════════════════════════════════════════╝
@st.cache_resource
def get_vector_stores():
    """Initialize three isolated ChromaDB collections for the memory system."""
    return init_databases(persist_dir=PERSIST_DIR, embedding_model=EMBEDDING_MODEL)


doc_db, chat_db, identity_db = get_vector_stores()


# ╔══════════════════════════════════════════════════════╗
# ║  2. SIDEBAR — DOCUMENT INGESTION PIPELINE            ║
# ╚══════════════════════════════════════════════════════╝

st.sidebar.header("🧠 Knowledge Base")
st.sidebar.caption("Upload documents (PDF, TXT, LOG) to build factual long-term memory.")

uploaded_file = st.sidebar.file_uploader("Choose a file...", type=["pdf", "txt", "log"])

# Session-level dedup tracking
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

# Auto-clear the vault-cleared flag (prevents re-ingestion for one rerun cycle)
if st.session_state.get("_just_cleared_vault"):
    del st.session_state["_just_cleared_vault"]
    uploaded_file = None  # Suppress the uploader for this rerun

if uploaded_file is not None and not st.session_state.get("_just_cleared_vault"):
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"

    # Check persistent dedup (same filename already in vault)
    already_ingested = False
    try:
        existing = doc_db._collection.get(where={"source": uploaded_file.name}, limit=1)
        if existing and existing["ids"]:
            already_ingested = True
    except Exception:
        pass

    if file_key in st.session_state.ingested_files or already_ingested:
        st.sidebar.info(f"**{uploaded_file.name}** is already in your knowledge base.")
    else:
        with st.sidebar.status("Processing document...", expanded=True) as status:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                documents = []

                if uploaded_file.name.lower().endswith(".pdf"):
                    # Detect if this is a tabular/structured document that needs Vision OCR
                    # even if it has digital text (PyMuPDF mangles table layouts)
                    _force_vision = bool(_TABULAR_DOC_PATTERNS.search(uploaded_file.name))

                    used_vision = False

                    if _force_vision:
                        # === FORCE VISION for tabular docs ===
                        st.write("📊 Tabular document detected — using AI Vision for accurate table extraction...")
                        progress_bar = st.progress(0)

                        def update_progress(current, total):
                            progress_bar.progress(current / total)
                            st.write(f"  📄 Processing page {current}/{total}...")

                        vision_text = extract_text_gemini_vision(
                            tmp_file_path, progress_callback=update_progress
                        )
                        if vision_text and len(vision_text.strip()) > 50:
                            st.write("✅ AI Vision OCR complete — tables preserved.")
                            documents = [
                                Document(
                                    page_content=vision_text,
                                    metadata={
                                        "source": uploaded_file.name,
                                        "extraction_method": "gemini_vision",
                                    },
                                )
                            ]
                            used_vision = True

                    if not used_vision:
                        # === TIER 1: Digital text extraction ===
                        st.write("⚡ Attempting fast digital extraction...")
                        digital_text = extract_text_digital(tmp_file_path)

                        if digital_text and len(digital_text.strip()) > 50:
                            st.write("✅ Digital PDF — text extracted.")
                            documents = [
                                Document(
                                    page_content=digital_text,
                                    metadata={
                                        "source": uploaded_file.name,
                                        "extraction_method": "pymupdf_digital",
                                    },
                                )
                            ]
                        else:
                            # === TIER 2: Gemini Vision OCR ===
                            st.write("📸 Scanned/complex document detected — engaging AI Vision OCR...")
                            progress_bar = st.progress(0)

                            def update_progress(current, total):
                                progress_bar.progress(current / total)
                                st.write(f"  📄 Processing page {current}/{total}...")

                            vision_text = extract_text_gemini_vision(
                                tmp_file_path, progress_callback=update_progress
                            )

                            if vision_text and len(vision_text.strip()) > 50:
                                st.write("✅ AI Vision OCR complete — tables preserved.")
                                documents = [
                                    Document(
                                        page_content=vision_text,
                                        metadata={
                                            "source": uploaded_file.name,
                                            "extraction_method": "gemini_vision",
                                        },
                                    )
                                ]
                            else:
                                # === TIER 3: RapidOCR fallback ===
                                st.write("⚠️ Vision OCR unavailable — falling back to local OCR...")
                                fallback_text = extract_text_rapidocr_fallback(tmp_file_path)
                                if fallback_text:
                                    documents = [
                                        Document(
                                            page_content=fallback_text,
                                            metadata={
                                                "source": uploaded_file.name,
                                                "extraction_method": "rapidocr_fallback",
                                            },
                                        )
                                    ]
                else:
                    # Text / Log files
                    loader = TextLoader(tmp_file_path, encoding="utf-8")
                    documents = loader.load()
                    for d in documents:
                        d.metadata["source"] = uploaded_file.name
                        d.metadata["extraction_method"] = "text_direct"

                if documents and len(documents[0].page_content.strip()) > 0:
                    # Smart chunking — tables preserved as atomic units
                    chunks = smart_chunk_documents(documents)
                    doc_db.add_documents(chunks)

                    table_chunks = sum(
                        1 for c in chunks if c.metadata.get("content_type") == "table"
                    )
                    text_chunks = len(chunks) - table_chunks

                    status.update(label="✅ Document ingested!", state="complete")
                    st.sidebar.success(
                        f"Memorized **{len(chunks)}** chunks from **{uploaded_file.name}**\n\n"
                        f"📝 Text chunks: {text_chunks}  •  📊 Table chunks: {table_chunks}"
                    )
                    st.session_state.ingested_files.add(file_key)

                    # === DEFER personal fact mining until LLM is initialized ===
                    # (llm is defined in section 3, after this sidebar block)
                    st.session_state["_pending_doc_facts"] = {
                        "text": documents[0].page_content,
                        "source": uploaded_file.name,
                    }
                else:
                    status.update(label="❌ No text found", state="error")
                    st.sidebar.error("Could not extract any text from this file.")

            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.sidebar.error(f"Error processing file: {e}")

            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)


# ── Web Access Toggle + URL Ingestor ──
st.sidebar.markdown("---")
st.sidebar.header("🌐 Web Access")

_web_enabled = st.sidebar.toggle(
    "Enable Web Search",
    value=True,
    help="When enabled, the AI can search the web for real-world info. "
         "Only triggers on questions about current events, general knowledge, etc. "
         "Personal data queries always use your local vault.",
)
if not DDGS_AVAILABLE:
    st.sidebar.caption("⚠️ Install `duckduckgo-search` for web access: `pip install duckduckgo-search`")

# URL Ingestor — paste a URL to save its content into the vault
with st.sidebar.expander("📎 Ingest a URL"):
    _url_input = st.text_input(
        "Paste a URL to save its content:",
        placeholder="https://example.com/article",
        key="url_ingest_input",
    )
    if st.button("📥 Fetch & Save", key="btn_url_ingest") and _url_input:
        with st.spinner("Fetching URL..."):
            _url_text = _fetch_url_text(_url_input)
            if _url_text and len(_url_text.strip()) > 50:
                _url_doc = Document(
                    page_content=_url_text,
                    metadata={
                        "source": _url_input[:100],
                        "extraction_method": "url_fetch",
                        "content_type": "text",
                    },
                )
                _url_chunks = smart_chunk_documents([_url_doc])
                doc_db.add_documents(_url_chunks)
                st.success(f"✅ Saved **{len(_url_chunks)}** chunks from URL")
            else:
                st.error("Could not extract meaningful text from that URL.")


# ╔══════════════════════════════════════════════════════╗
# ║  3. MODEL SELECTION                                  ║
# ╚══════════════════════════════════════════════════════╝

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Model Settings")
llm_choice = st.sidebar.radio(
    "Select AI Brain:",
    ["Local — Gemma 2 9B (recommended)", "Local — Llama 3 8B", "Cloud API (Gemini Flash)"],
    help="Local requires Ollama running. Cloud uses your Google API key.",
)

_LOCAL_MODEL_MAP = {
    "Local — Gemma 2 9B (recommended)": "gemma2:9b",
    "Local — Llama 3 8B": "llama3",
}

if llm_choice in _LOCAL_MODEL_MAP:
    try:
        llm = ChatOllama(model=_LOCAL_MODEL_MAP[llm_choice])
    except Exception:
        st.error("⚠️ Cannot connect to Ollama. Make sure it is running (`ollama serve`).")
        st.stop()
else:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        timeout=30,
        max_retries=2,
    )

# ── Dedicated extraction LLM: always Gemini 2.5 Flash Lite for structured JSON ──
# Why 2.5-flash-lite (not 2.5-flash)?  Free-tier quota:
#   gemini-2.5-flash      = 20 RPD  (shared with chat → exhausted fast)
#   gemini-2.5-flash-lite = separate, generous quota pool
# Chat LLM handles the conversation (local = private & free).
# Extraction runs in background threads — API latency is invisible.
_api_key = os.getenv("GOOGLE_API_KEY")
if _api_key:
    extraction_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        timeout=30,
        max_retries=2,
    )
else:
    # No API key — fall back to whatever chat model is selected
    extraction_llm = llm


# ── Process deferred document fact extraction (uses extraction_llm for JSON reliability) ──
if "_pending_doc_facts" in st.session_state:
    _pending = st.session_state.pop("_pending_doc_facts")
    try:
        _doc_facts = extract_facts_from_document(
            _pending["text"], _pending["source"], identity_db, extraction_llm
        )
        if _doc_facts > 0:
            st.sidebar.info(f"🧬 Extracted **{_doc_facts}** personal fact(s) from **{_pending['source']}**.")
    except Exception as e:
        print(f"[Deferred Doc Fact Extraction Error] {e}")


# ╔══════════════════════════════════════════════════════╗
# ║  4. TRIPLE-MEMORY UNIFIED PROMPT                     ║
# ╚══════════════════════════════════════════════════════╝

# Dynamically route the prompt based on the sidebar toggle
if "Local" in llm_choice:
    prompt_template = ChatPromptTemplate.from_template(_local_prompt_text)
else:
    prompt_template = ChatPromptTemplate.from_template(_cloud_prompt_text)


# ╔══════════════════════════════════════════════════════╗
# ║  5. MEMORY DASHBOARD & MANAGEMENT                   ║
# ╚══════════════════════════════════════════════════════╝

st.sidebar.markdown("---")
st.sidebar.header("🔍 Memory Dashboard")

# Live memory statistics
try:
    doc_count = doc_db._collection.count()
    chat_count = chat_db._collection.count()
    identity_count = identity_db._collection.count()
except Exception:
    doc_count = chat_count = identity_count = 0

col1, col2, col3 = st.sidebar.columns(3)
col1.metric("📄 Docs", doc_count)
col2.metric("💬 Chats", chat_count)
col3.metric("🧬 Facts", identity_count)


# --- Memory Audit ---
if st.sidebar.button("🔎 Full Memory Audit"):
    audit_question = (
        "Compile a comprehensive dossier of everything you know about me: "
        "identity, education, skills, projects, tech preferences, personality traits, "
        "coding style, and any other personal details. Cite the source of each piece "
        "of information (document name or conversation). Only use stored memory. "
        "Do not fabricate anything."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.session_state.messages.append(
        {"role": "user", "content": "🔎 Run a full memory audit — what do you know about me?"}
    )

    with st.spinner("Scanning all three memory banks..."):
        docs = retrieve_from_vault(audit_question, doc_db, default_k=8)
        chats = chat_db.similarity_search(audit_question, k=5)
        identities = identity_db.similarity_search(audit_question, k=15)

        doc_context = "\n\n".join(
            [f"[Source: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs]
        )
        chat_context = "\n\n".join([c.page_content for c in chats])
        identity_context = "\n".join(
            [
                f"• [{i.metadata.get('category', 'general')}] {i.page_content}"
                for i in identities
            ]
        )

        chain = prompt_template | llm
        response = chain.invoke(
            {
                "doc_context": doc_context,
                "chat_context": chat_context,
                "identity_context": identity_context,
                "web_context": "(No web search performed)",
                "recent_messages": "(Memory audit requested)",
                "current_datetime": datetime.now().strftime("%A, %B %d, %Y at %I:%M %p"),
                "clarification_note": "",
                "user_input": audit_question,
            }
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": response.content}
        )
    st.rerun()


# --- Browse Identity Facts ---
with st.sidebar.expander("🧬 View Known Facts About You"):
    try:
        all_facts = identity_db._collection.get(limit=50, include=["documents", "metadatas"])
        if all_facts and all_facts["documents"]:
            for doc_text, meta in zip(all_facts["documents"], all_facts["metadatas"]):
                cat = meta.get("category", "general")
                conf = meta.get("confidence", "?")
                _conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
                st.markdown(f"{_conf_icon} **[{cat}]** {doc_text}")
        else:
            st.caption("_No personal facts learned yet. Start chatting!_")
    except Exception:
        st.caption("_Memory initializing..._")

    # Show pending clarifications count
    _pending_q = st.session_state.get("_clarification_queue", [])
    if _pending_q:
        st.caption(f"⚠️ {len(_pending_q)} fact(s) pending your clarification")


# --- Memory Management ---

# Show feedback from previous clear action (persists across rerun)
if "clear_feedback" in st.session_state:
    st.sidebar.success(st.session_state.clear_feedback)
    del st.session_state.clear_feedback

with st.sidebar.expander("⚠️ Memory Management"):
    st.caption("Clear specific memory banks. This cannot be undone.")

    if st.button("🗑️ Clear Chat Memory", key="btn_clear_chat"):
        _clear_collection(chat_db._collection, "Episodic memory")

    if st.button("🗑️ Clear Identity Facts", key="btn_clear_identity"):
        _clear_collection(identity_db._collection, "Identity facts")

    if st.button("🗑️ Clear Document Vault", key="btn_clear_docs"):
        st.session_state.ingested_files = set()
        st.session_state["_just_cleared_vault"] = True
        _clear_collection(doc_db._collection, "Document vault")

    # --- Per-document removal ---
    st.markdown("---")
    st.caption("Remove a single document:")
    try:
        _vault_meta = doc_db._collection.get(limit=5000, include=["metadatas"])
        _vault_sources = sorted(
            {m["source"] for m in (_vault_meta.get("metadatas") or []) if "source" in m}
        )
    except Exception:
        _vault_sources = []

    if _vault_sources:
        _doc_to_remove = st.selectbox(
            "Select document", _vault_sources, key="sel_doc_remove"
        )
        if st.button(f"🗑️ Remove \"{_doc_to_remove}\"", key="btn_remove_single_doc"):
            try:
                _hit = doc_db._collection.get(
                    where={"source": _doc_to_remove}, limit=500
                )
                if _hit and _hit["ids"]:
                    doc_db._collection.delete(ids=_hit["ids"])
                    # Clear session dedup cache entries for this file
                    st.session_state.ingested_files = {
                        fk for fk in st.session_state.get("ingested_files", set())
                        if not fk.startswith(_doc_to_remove.rsplit(".", 1)[0])
                    }
                    st.session_state["_just_cleared_vault"] = True
                    st.session_state.clear_feedback = (
                        f"✅ Removed **{_doc_to_remove}** ({len(_hit['ids'])} chunks)."
                    )
                    st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error removing document: {e}")
    else:
        st.caption("_No documents in vault._")


# ╔══════════════════════════════════════════════════════╗
# ║  6. CHAT INTERFACE — TRIPLE-MEMORY RETRIEVAL LOOP   ║
# ╚══════════════════════════════════════════════════════╝

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Show extraction feedback from background thread (if any)
_extraction_msg = st.session_state.pop("_last_extraction_result", None)
if _extraction_msg:
    st.toast(_extraction_msg, icon="🧬")

if user_input := st.chat_input("What is on your mind?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── Process any pending clarification the user might be answering ──
    if st.session_state.get("_awaiting_clarification"):
        _process_clarification_response(user_input, identity_db, extraction_llm)
        st.session_state["_awaiting_clarification"] = False

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # ── Retrieve from all three memory banks ──
            # Source-aware: if user asks about "my resume", gets ALL resume chunks
            docs = retrieve_from_vault(user_input, doc_db)

            # Detect meta-queries about recency or memory recall
            _is_recency_query = bool(_RECENCY_META_PATTERNS.search(user_input))
            _is_memory_recall = bool(_MEMORY_RECALL_PATTERNS.search(user_input))

            if _is_recency_query:
                # User asks "what did we talk about recently?" → pull by timestamp, not topic
                chats = _get_recent_episodic_memories(chat_db, k=12)
            else:
                chats = _recency_boosted_episodic_search(user_input, chat_db, k=8)

            if _is_memory_recall:
                # User says "you forgot" / "I told you" → pull ALL identity facts
                identities = _get_all_identity_facts(identity_db)
                # Also include recent episodic memories for context
                if not _is_recency_query:
                    chats = _get_recent_episodic_memories(chat_db, k=12)
            else:
                identities = identity_db.similarity_search(user_input, k=10)

            # Detect casual/short messages — don't pollute with irrelevant context
            _is_casual = _is_casual_message(user_input)

            if _is_casual:
                doc_context = "(No documents relevant to this message)"
                chat_context = "(Casual conversation)"
            else:
                doc_context = "\n\n".join(
                    [
                        f"[Source: {d.metadata.get('source', '?')}]\n{d.page_content}"
                        for d in docs
                    ]
                ) if docs else ""
                chat_context = "\n\n".join(
                    [c.page_content for c in chats]
                ) if chats else ""

                # Guard: user asks about docs but retrieval returned nothing
                _lower = user_input.lower()
                _asks_about_docs = any(kw in _lower for kw in _DATA_KEYWORDS)
                if _asks_about_docs and not doc_context.strip():
                    doc_context = (
                        "[SYSTEM: No document content was found in the vault. "
                        "The user may not have uploaded the document yet, "
                        "or the query didn't match any stored chunks. "
                        "Do NOT fabricate any data — tell the user you don't have it.]"
                    )

            identity_context = "\n".join(
                [
                    f"• [{i.metadata.get('category', 'general')}] {i.page_content}"
                    for i in identities
                ]
            )

            # ── Build recent conversation window (last 10 turns, excluding current) ──
            recent_msgs = st.session_state.messages[-11:-1]  # Exclude the just-added user message
            recent_messages = "\n".join(
                [f"{'Human' if m['role'] == 'user' else 'AI'}: {m['content'][:800]}" for m in recent_msgs]
            ) if recent_msgs else "(First message in this session)"

            # ── Web search: auto-detect OR follow-up on previous results ──
            web_context = "(No web search performed)"
            _followup_content = _check_web_followup(user_input) if not _is_casual else None
            if _followup_content:
                web_context = _followup_content
                # Suppress vault docs so the LLM focuses on the web article
                doc_context = "(No documents relevant — answering from web article)"
                chat_context = ""
            elif _web_enabled and not _is_casual and _is_web_query(user_input):
                _web_results = _search_web(user_input, max_results=5)
                if _web_results:
                    web_context = f"[Web search for: \"{user_input}\"]\n\n{_web_results}"

            # ── Check for pending clarifications to append ──
            clarification_note = ""
            if not _is_casual:
                _clarification_text = _build_clarification_prompt()
                if _clarification_text:
                    clarification_note = (
                        "\n══ CLARIFICATION REQUEST ══\n"
                        "After answering the user's message, append the following question "
                        "naturally at the end of your response:\n"
                        f"{_clarification_text}\n"
                    )
                    st.session_state["_awaiting_clarification"] = True

            # ── Stream the response ──
            chain = prompt_template | llm

            full_response = ""
            response_placeholder = st.empty()
            try:
                for chunk in chain.stream(
                    {
                        "doc_context": doc_context,
                        "chat_context": chat_context,
                        "identity_context": identity_context,
                        "web_context": web_context,
                        "recent_messages": recent_messages,
                        "current_datetime": datetime.now().strftime("%A, %B %d, %Y at %I:%M %p"),
                        "clarification_note": clarification_note,
                        "user_input": user_input,
                    }
                ):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
            except Exception as stream_err:
                # Fallback to non-streaming if streaming fails
                try:
                    response_message = chain.invoke(
                        {
                            "doc_context": doc_context,
                            "chat_context": chat_context,
                            "identity_context": identity_context,
                            "web_context": web_context,
                            "recent_messages": recent_messages,
                            "current_datetime": datetime.now().strftime("%A, %B %d, %Y at %I:%M %p"),
                            "clarification_note": clarification_note,
                            "user_input": user_input,
                        }
                    )
                    full_response = response_message.content
                    response_placeholder.markdown(full_response)
                except Exception as invoke_err:
                    error_msg = str(invoke_err)
                    if "429" in error_msg or "quota" in error_msg.lower():
                        full_response = "⚠️ API quota exceeded. Either wait a few minutes or switch to **Local Offline (Llama 3)** in the sidebar."
                    elif "timeout" in error_msg.lower():
                        full_response = "⚠️ Request timed out. The API might be overloaded — try again or switch to local mode."
                    else:
                        full_response = f"⚠️ Error getting response: {error_msg[:200]}"
                    response_placeholder.markdown(full_response)

            response_text = full_response

    st.session_state.messages.append({"role": "assistant", "content": response_text})

    # ── ORGANIC LEARNING: Save to Episodic Memory (split by role for clean retrieval) ──
    _ts_iso = datetime.now().isoformat()
    _ts_human = datetime.now().strftime("%Y-%m-%d %H:%M")
    _turn_id = f"turn_{int(time.time())}"

    # Store user message and AI response as SEPARATE chunks with shared turn_id
    # This lets retrieval find relevant conversations without AI filler noise
    chat_db.add_texts(
        [
            f"[{_ts_human}] User: {user_input}",
            f"[{_ts_human}] AI: {response_text}",
        ],
        metadatas=[
            {
                "timestamp": _ts_iso,
                "type": "conversation",
                "role": "user",
                "turn_id": _turn_id,
                "content_preview": user_input[:300],
            },
            {
                "timestamp": _ts_iso,
                "type": "conversation",
                "role": "assistant",
                "turn_id": _turn_id,
                "content_preview": response_text[:300],
            },
        ],
    )

    # ── PASSIVE IDENTITY EXTRACTION (NON-BLOCKING THREAD via Gemini Flash) ──
    bg_thread = threading.Thread(
        target=run_background_extraction,
        args=(user_input, response_text, identity_db, extraction_llm, "Cloud" in llm_choice),
    )
    add_script_run_ctx(bg_thread)
    bg_thread.start()

    # ── AUTO-SUMMARIZE SESSION (periodic consolidation) ──
    _maybe_summarize_session(st.session_state.messages, chat_db, extraction_llm)

    # ── PERIODIC STALENESS CHECK (flags old facts for re-verification) ──
    _check_stale_facts(identity_db)
