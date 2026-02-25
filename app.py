import streamlit as st
import os
import re
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import tempfile
import fitz  # PyMuPDF

# LangChain Imports
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# Gemini Vision (for table-aware OCR) — uses modern google-genai SDK
try:
    from google import genai as google_genai
    GEMINI_VISION_AVAILABLE = True
except ImportError:
    GEMINI_VISION_AVAILABLE = False

load_dotenv()

# ╔══════════════════════════════════════════════════════╗
# ║              CONFIGURATION                           ║
# ╚══════════════════════════════════════════════════════╝
PAGE_TITLE = "Personal AI Agent"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = "./ai_memory_db"
GEMINI_VISION_MODEL = "gemini-2.0-flash"

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(f"🧠 {PAGE_TITLE}")


# ╔══════════════════════════════════════════════════════╗
# ║  1. TRIPLE-MEMORY ARCHITECTURE (ChromaDB)            ║
# ╚══════════════════════════════════════════════════════╝
@st.cache_resource
def get_vector_stores():
    """Initialize three isolated ChromaDB collections for the memory system."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Collection A: THE VAULT — Strictly for uploaded documents (PDF, TXT, LOG)
    doc_db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="document_vault",
    )

    # Collection B: EPISODIC MEMORY — Every chat turn is saved here
    chat_db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="episodic_memory",
    )

    # Collection C: CORE IDENTITY — Extracted personal facts & preferences
    # This is the "personality crystal" that grows over time.
    identity_db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="core_identity",
    )

    return doc_db, chat_db, identity_db


doc_db, chat_db, identity_db = get_vector_stores()


# ╔══════════════════════════════════════════════════════╗
# ║  2. THREE-TIER OCR ENGINE                            ║
# ║     Tier 1: PyMuPDF digital text (instant)           ║
# ║     Tier 2: Gemini Vision — table-aware (best)       ║
# ║     Tier 3: RapidOCR local fallback (last resort)    ║
# ╚══════════════════════════════════════════════════════╝

VISION_OCR_PROMPT = """You are a precision document OCR engine. Extract ALL text from this document page.

CRITICAL RULES:
1. **Tables**: Reproduce them EXACTLY in Markdown table format:
   | Column1 | Column2 | Column3 |
   |---------|---------|---------|
   | data    | data    | data    |
2. **Marksheets/Scorecards**: Each subject MUST stay connected to its correct marks/grade on the SAME row. Never mix rows.
3. **Ignore visual noise**: Skip watermarks, page borders, background patterns, stamps, logos, and decorative elements entirely.
4. **Preserve hierarchy**: Maintain headings, subheadings, and section structure with Markdown (## , ### ).
5. **Reading order**: Follow logical top-to-bottom, left-to-right reading order.
6. **Numerical precision**: Be extremely precise with all numbers — scores, dates, roll numbers, totals, percentages.
7. If a section is genuinely illegible, write [ILLEGIBLE] instead of guessing.

Output clean Markdown. No commentary — just the extracted content."""


def _init_gemini_vision():
    """Initialize Gemini Vision client if API key is present."""
    if not GEMINI_VISION_AVAILABLE:
        return None
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        return google_genai.Client(api_key=api_key)
    except Exception:
        return None


def extract_text_digital(pdf_path):
    """TIER 1: Fast digital text extraction via PyMuPDF (layout-aware)."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text", sort=True)
        if text.strip():
            pages.append(f"## Page {page_num + 1}\n\n{text}")
    doc.close()
    return "\n\n".join(pages)


def extract_text_gemini_vision(pdf_path, progress_callback=None):
    """TIER 2: Gemini Vision OCR — understands tables, ignores noise."""
    client = _init_gemini_vision()
    if not client:
        return None

    doc = fitz.open(pdf_path)
    all_pages = []

    for page_num in range(len(doc)):
        if progress_callback:
            progress_callback(page_num + 1, len(doc))

        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)  # High res for accuracy
        img_bytes = pix.tobytes("png")

        try:
            from google.genai import types as genai_types

            image_part = genai_types.Part.from_bytes(
                data=img_bytes, mime_type="image/png"
            )
            response = client.models.generate_content(
                model=GEMINI_VISION_MODEL,
                contents=[image_part, VISION_OCR_PROMPT],
            )
            page_text = response.text if response.text else ""
            all_pages.append(f"## Page {page_num + 1}\n\n{page_text}")

            # Gentle rate limit to avoid API throttling
            if page_num < len(doc) - 1:
                time.sleep(0.5)
        except Exception as e:
            all_pages.append(f"## Page {page_num + 1}\n\n[Vision OCR Error: {e}]")

    doc.close()
    return "\n\n---\n\n".join(all_pages)


def extract_text_rapidocr_fallback(pdf_path):
    """TIER 3: Local RapidOCR — last resort when offline & scanned."""
    try:
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()
        doc = fitz.open(pdf_path)
        pages = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            result, _ = ocr(img_bytes)
            if result:
                page_text = "\n".join([line[1] for line in result])
                pages.append(f"## Page {page_num + 1}\n\n{page_text}")
        doc.close()
        return "\n\n".join(pages)
    except Exception:
        return ""


# ╔══════════════════════════════════════════════════════╗
# ║  3. TABLE-PRESERVING INTELLIGENT CHUNKER             ║
# ╚══════════════════════════════════════════════════════╝

def smart_chunk_documents(documents):
    """
    Split documents while keeping Markdown tables as atomic chunks.
    Tables are NEVER split mid-row — they stay intact for clean vector retrieval.
    """
    all_chunks = []

    for doc in documents:
        content = doc.page_content
        base_meta = doc.metadata.copy()

        # Regex: match complete markdown table blocks (header + separator + rows)
        table_pattern = r'(\|[^\n]+\|\n(?:\|[\s:|-]+\|\n)(?:\|[^\n]+\|\n?)*)'
        parts = re.split(table_pattern, content)

        chunk_docs = []
        text_buf = ""

        for part in parts:
            is_table = bool(re.match(r'\s*\|[^\n]+\|', part.strip()))

            if is_table:
                # Flush accumulated plain text first
                if text_buf.strip():
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200
                    )
                    for tc in splitter.split_text(text_buf):
                        chunk_docs.append(
                            Document(
                                page_content=tc,
                                metadata={**base_meta, "content_type": "text"},
                            )
                        )
                    text_buf = ""

                # Large table? Split by row batches, preserve header
                table_text = part.strip()
                if len(table_text) > 3000:
                    lines = table_text.split("\n")
                    header_lines = lines[:2]  # header + separator
                    data_lines = lines[2:]
                    header = "\n".join(header_lines)
                    batch_size = 20
                    for i in range(0, len(data_lines), batch_size):
                        batch = "\n".join(data_lines[i : i + batch_size])
                        chunk_docs.append(
                            Document(
                                page_content=f"{header}\n{batch}",
                                metadata={**base_meta, "content_type": "table"},
                            )
                        )
                else:
                    chunk_docs.append(
                        Document(
                            page_content=table_text,
                            metadata={**base_meta, "content_type": "table"},
                        )
                    )
            else:
                text_buf += part

        # Flush remaining text
        if text_buf.strip():
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            for tc in splitter.split_text(text_buf):
                chunk_docs.append(
                    Document(
                        page_content=tc,
                        metadata={**base_meta, "content_type": "text"},
                    )
                )

        # Enrich every chunk with indexing metadata
        for i, chunk in enumerate(chunk_docs):
            chunk.metadata.update(
                {
                    "chunk_index": i,
                    "total_chunks": len(chunk_docs),
                    "ingested_at": datetime.now().isoformat(),
                }
            )

        all_chunks.extend(chunk_docs)

    return all_chunks


# ╔══════════════════════════════════════════════════════╗
# ║  4. PASSIVE IDENTITY EXTRACTION ENGINE               ║
# ║     Mines personal facts from every conversation     ║
# ║     and stores them in core_identity (deduped).      ║
# ╚══════════════════════════════════════════════════════╝

FACT_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """Analyze this conversation and extract any NEW personal facts about the human user.

<conversation>
Human: {user_input}
AI: {ai_response}
</conversation>

Extract facts in these categories:
- identity: name, age, location, occupation, education, background
- preferences: tech stack, tools, languages, frameworks, methodologies they prefer
- projects: what they're building, goals, deadlines, aspirations
- skills: what they know, what they're learning, expertise level
- opinions: views on technologies, approaches, philosophies
- personal: hobbies, interests, communication style

RULES:
- Only extract facts EXPLICITLY stated or strongly implied by the HUMAN's words.
- Do NOT extract facts about the AI or generic statements.
- Each fact must be a complete, self-contained sentence.
- If NO personal facts are present, return an empty array.

Return ONLY a valid JSON array (no markdown fences, no explanation):
[{{"category": "...", "fact": "..."}}]

If nothing found:
[]"""
)


def _jaccard_similarity(text1, text2):
    """Quick word-overlap similarity for deduplication."""
    w1 = set(text1.lower().split())
    w2 = set(text2.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def extract_personal_facts(user_input, ai_response, identity_db, llm):
    """
    Background extraction: mine personal facts from the conversation turn,
    deduplicate against existing identity store, and persist new ones.
    Returns the count of newly stored facts.
    """
    try:
        chain = FACT_EXTRACTION_PROMPT | llm
        result = chain.invoke(
            {"user_input": user_input, "ai_response": ai_response}
        )

        raw = result.content.strip()
        # Extract JSON array even if wrapped in markdown fences
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not json_match:
            return 0

        facts = json.loads(json_match.group())
        if not facts or not isinstance(facts, list):
            return 0

        stored = 0
        for fact in facts:
            if not isinstance(fact, dict) or "fact" not in fact:
                continue

            fact_text = fact["fact"].strip()
            if len(fact_text) < 5:
                continue

            # Deduplication: skip if a very similar fact already exists
            # If similar but not identical, UPDATE the old one (user may have changed preference)
            try:
                existing = identity_db.similarity_search(fact_text, k=1)
                if existing:
                    sim = _jaccard_similarity(existing[0].page_content, fact_text)
                    if sim > 0.85:
                        # Near-duplicate — skip entirely
                        continue
                    elif sim > 0.55:
                        # Similar topic but updated info — delete old, store new
                        try:
                            old_results = identity_db._collection.get(
                                where={"category": fact.get("category", "general")},
                                limit=10,
                                include=["documents"]
                            )
                            if old_results and old_results["documents"]:
                                for idx, old_doc in enumerate(old_results["documents"]):
                                    if _jaccard_similarity(old_doc, fact_text) > 0.55:
                                        identity_db._collection.delete(
                                            ids=[old_results["ids"][idx]]
                                        )
                                        break
                        except Exception:
                            pass
            except Exception:
                pass

            identity_db.add_texts(
                [fact_text],
                metadatas=[
                    {
                        "category": fact.get("category", "general"),
                        "extracted_at": datetime.now().isoformat(),
                        "source_snippet": user_input[:200],
                    }
                ],
            )
            stored += 1

        return stored
    except Exception as e:
        # Log extraction failures for debugging (visible in terminal)
        import traceback
        print(f"[Identity Extraction Error] {e}")
        traceback.print_exc()
        return 0


def extract_facts_from_document(doc_text, source_name, identity_db, llm):
    """
    Mine personal facts from an uploaded document (e.g. resume, transcript).
    Called once at ingestion time — not on every chat.
    """
    DOC_FACT_PROMPT = ChatPromptTemplate.from_template(
        """Extract personal facts about the document owner from this uploaded document.

<document source="{source_name}">
{doc_text}
</document>

Extract facts into these categories:
- identity: name, age, DOB, location, nationality, occupation, education, university, degree
- skills: programming languages, tools, frameworks, certifications, academic scores
- projects: project names, descriptions, technologies used
- preferences: stated preferences, tools used repeatedly
- personal: hobbies, interests, achievements, awards

RULES:
- Only extract facts that clearly belong to the document owner (not third parties).
- Each fact must be a complete, standalone sentence.
- Be specific with numbers: GPA, scores, percentages, dates.
- If this is a marksheet, extract: student name, institution, subjects with scores, total/percentage.
- If this is a resume, extract: name, education history, skills, project summaries.
- If NO personal facts are present, return an empty array.

Return ONLY a valid JSON array:
[{{"category": "...", "fact": "..."}}]

If nothing found:
[]"""
    )

    try:
        # Take first 3000 chars to stay within token limits
        trimmed = doc_text[:3000]
        chain = DOC_FACT_PROMPT | llm
        result = chain.invoke({"doc_text": trimmed, "source_name": source_name})

        raw = result.content.strip()
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not json_match:
            return 0

        facts = json.loads(json_match.group())
        if not facts or not isinstance(facts, list):
            return 0

        stored = 0
        for fact in facts:
            if not isinstance(fact, dict) or "fact" not in fact:
                continue
            fact_text = fact["fact"].strip()
            if len(fact_text) < 5:
                continue

            # Dedup against existing facts
            try:
                existing = identity_db.similarity_search(fact_text, k=1)
                if existing and _jaccard_similarity(existing[0].page_content, fact_text) > 0.75:
                    continue
            except Exception:
                pass

            identity_db.add_texts(
                [fact_text],
                metadatas=[{
                    "category": fact.get("category", "general"),
                    "extracted_at": datetime.now().isoformat(),
                    "source_document": source_name,
                }],
            )
            stored += 1
        return stored
    except Exception as e:
        print(f"[Document Fact Extraction Error] {e}")
        return 0


# ╔══════════════════════════════════════════════════════╗
# ║  4b. SOURCE-AWARE DOCUMENT RETRIEVAL                 ║
# ║      When user asks about "my resume", fetch ALL     ║
# ║      chunks from that doc — not just top-k.          ║
# ╚══════════════════════════════════════════════════════╝

def _match_document_source(query, sources):
    """Check if the user's query references a known uploaded document."""
    if not sources:
        return None
    query_lower = query.lower()

    # Direct filename match (e.g. "arjun_resume" in query)
    for src in sources:
        stem = src.lower().rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
        if stem in query_lower or src.lower() in query_lower:
            return src

    # Type-keyword match (e.g. user says "resume" and we have "ArjunYS_Resume.pdf")
    DOC_TYPE_KEYWORDS = {
        "resume":      ["resume", "cv", "curriculum vitae"],
        "marksheet":   ["marksheet", "marks", "transcript", "grade", "grades",
                        "scorecard", "score card", "result"],
        "certificate": ["certificate", "certification", "cert"],
        "report":      ["report", "semester"],
    }
    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            # Find a source whose filename matches this type
            for src in sources:
                src_lower = src.lower()
                if doc_type in src_lower or any(kw in src_lower for kw in keywords):
                    return src
            break  # Matched keyword but no source filename — stop

    return None


# Broad queries that mean "dump everything you know" — need aggressive retrieval
_BROAD_QUERY_PATTERNS = re.compile(
    r"\b(list.*(everything|all)|everything you know|tell me everything|what do you know"
    r"|all about me|full (audit|summary|overview|recap)|summarize|what.*have.*on me"
    r"|show me everything|what.*remember)\b",
    re.IGNORECASE,
)


def retrieve_from_vault(query, doc_db, default_k=5):
    """
    Smart document retrieval:
    - Broad queries ("list out everything") → fetch ALL docs from vault.
    - Source-specific queries ("my resume") → fetch ALL chunks from that doc.
    - Otherwise → standard similarity search with default_k results.
    """
    # Collect all known source filenames in the vault
    try:
        meta_results = doc_db._collection.get(limit=5000, include=["metadatas"])
        sources = {
            m["source"] for m in (meta_results.get("metadatas") or []) if "source" in m
        }
    except Exception:
        sources = set()

    # ── Broad query: dump everything from the vault ──
    if _BROAD_QUERY_PATTERNS.search(query):
        try:
            results = doc_db._collection.get(
                limit=200,
                include=["documents", "metadatas"],
            )
            if results and results["documents"]:
                return [
                    Document(page_content=text, metadata=meta)
                    for text, meta in zip(results["documents"], results["metadatas"])
                ]
        except Exception:
            pass

    # ── Source-specific query: fetch all chunks from matched doc ──
    matched_source = _match_document_source(query, sources)

    if matched_source:
        try:
            results = doc_db._collection.get(
                where={"source": matched_source},
                limit=100,
                include=["documents", "metadatas"],
            )
            if results and results["documents"]:
                return [
                    Document(page_content=text, metadata=meta)
                    for text, meta in zip(results["documents"], results["metadatas"])
                ]
        except Exception:
            pass

    # Default: similarity search
    return doc_db.similarity_search(query, k=default_k)


# ╔══════════════════════════════════════════════════════╗
# ║  5. SIDEBAR — DOCUMENT INGESTION PIPELINE            ║
# ╚══════════════════════════════════════════════════════╝

st.sidebar.header("🧠 Knowledge Base")
st.sidebar.caption("Upload documents (PDF, TXT, LOG) to build factual long-term memory.")

uploaded_file = st.sidebar.file_uploader("Choose a file...", type=["pdf", "txt", "log"])

# Session-level dedup tracking
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

if uploaded_file is not None:
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
                    # (llm is defined in section 6, after this sidebar block)
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


# ╔══════════════════════════════════════════════════════╗
# ║  6. MODEL SELECTION                                  ║
# ╚══════════════════════════════════════════════════════╝

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Model Settings")
llm_choice = st.sidebar.radio(
    "Select AI Brain:",
    ["Local Offline (Llama 3)", "Cloud API (Gemini Flash)"],
    help="Local requires Ollama running. Cloud uses your Google API key.",
)

if llm_choice == "Local Offline (Llama 3)":
    try:
        llm = ChatOllama(model="llama3")
    except Exception:
        st.error("⚠️ Cannot connect to Ollama. Make sure it is running (`ollama serve`).")
        st.stop()
else:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        timeout=30,          # 30s hard timeout — prevents infinite retry on 429
        max_retries=2,       # Retry twice, then fail gracefully
    )


# ── Process deferred document fact extraction (now that llm exists) ──
if "_pending_doc_facts" in st.session_state:
    _pending = st.session_state.pop("_pending_doc_facts")
    try:
        _doc_facts = extract_facts_from_document(
            _pending["text"], _pending["source"], identity_db, llm
        )
        if _doc_facts > 0:
            st.sidebar.info(f"🧬 Extracted **{_doc_facts}** personal fact(s) from **{_pending['source']}**.")
    except Exception as e:
        print(f"[Deferred Doc Fact Extraction Error] {e}")


# ╔══════════════════════════════════════════════════════╗
# ║  7. TRIPLE-MEMORY UNIFIED PROMPT                     ║
# ╚══════════════════════════════════════════════════════╝

prompt_template = ChatPromptTemplate.from_template(
    """You are the user's personal AI — a second brain, not a chatbot. You know them and grow with them.

# WHO THIS HUMAN IS
{identity_context}

# THEIR DOCUMENTS
{doc_context}

# THINGS YOU'VE TALKED ABOUT BEFORE
{chat_context}

# RECENT CONVERSATION
{recent_messages}

# RESPONSE RULES (MANDATORY — VIOLATING THESE IS A FAILURE)
1. ABSOLUTELY NO cheerleading, hype, or compliments on their work. Never say "impressive", "fascinating", "great", "amazing", "awesome", "incredible", "That's some impressive work!", or any variant. You are not their cheerleader.
2. NEVER open with filler. No "So,", "Well,", "Let me see...", "Great question!", "Sure!", "Of course!". Start with the actual answer.
3. NEVER close with offers. No "Let me know if you need anything else!", "Feel free to ask!", "That's about it for now!". Just stop.
4. NEVER narrate your own actions. No "Let me take a look...", "I remember that...", "I can see that...". Just state the facts.
5. NEVER announce what you know about them. Don't say "since you prefer Python" or "As I recall". Just use the knowledge silently.
6. Be COMPLETE. When listing things (projects, skills, scores), list ALL of them from the context. Don't summarize or skip items. If there are 3 projects, list 3. If there are 10 subjects, list 10.
7. Be DIRECT. Short answers for simple questions. Detailed answers for complex ones. No padding.
8. Use their name sparingly — only where it'd be natural in real speech.
9. If you don't know something, say "I don't have that" in one sentence. Don't hedge.
10. Markdown tables for tabular data. Code blocks for code. No decorative formatting.
11. Document facts override conversation memory.
12. NEVER fabricate personal details. If it's not in the context above, you don't know it.

Human: {user_input}
AI:"""
)


# ╔══════════════════════════════════════════════════════╗
# ║  8. MEMORY DASHBOARD & MANAGEMENT                   ║
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
                "recent_messages": "(Memory audit requested)",
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
                st.markdown(f"**[{cat}]** {doc_text}")
        else:
            st.caption("_No personal facts learned yet. Start chatting!_")
    except Exception:
        st.caption("_Memory initializing..._")


# --- Memory Management ---

# Show feedback from previous clear action (persists across rerun)
if "clear_feedback" in st.session_state:
    st.sidebar.success(st.session_state.clear_feedback)
    del st.session_state.clear_feedback

def _clear_collection(collection, label):
    """Delete all documents from a ChromaDB collection in batches."""
    try:
        total_deleted = 0
        while True:
            result = collection.get(limit=5000)
            ids = result["ids"]
            if not ids:
                break
            collection.delete(ids=ids)
            total_deleted += len(ids)
        if total_deleted > 0:
            st.session_state.clear_feedback = f"✅ {label} cleared ({total_deleted} entries removed)."
        else:
            st.session_state.clear_feedback = f"ℹ️ {label} is already empty."
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error clearing {label}: {e}")

with st.sidebar.expander("⚠️ Memory Management"):
    st.caption("Clear specific memory banks. This cannot be undone.")

    if st.button("🗑️ Clear Chat Memory", key="btn_clear_chat"):
        _clear_collection(chat_db._collection, "Episodic memory")

    if st.button("🗑️ Clear Identity Facts", key="btn_clear_identity"):
        _clear_collection(identity_db._collection, "Identity facts")

    if st.button("🗑️ Clear Document Vault", key="btn_clear_docs"):
        _clear_collection(doc_db._collection, "Document vault")


# ╔══════════════════════════════════════════════════════╗
# ║  9. CHAT INTERFACE — TRIPLE-MEMORY RETRIEVAL LOOP   ║
# ╚══════════════════════════════════════════════════════╝

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("What is on your mind?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # ── Retrieve from all three memory banks ──
            # Source-aware: if user asks about "my resume", gets ALL resume chunks
            docs = retrieve_from_vault(user_input, doc_db)
            chats = chat_db.similarity_search(user_input, k=3)
            identities = identity_db.similarity_search(user_input, k=5)

            doc_context = "\n\n".join(
                [
                    f"[Source: {d.metadata.get('source', '?')}]\n{d.page_content}"
                    for d in docs
                ]
            )
            chat_context = "\n\n".join([c.page_content for c in chats])
            identity_context = "\n".join(
                [
                    f"• [{i.metadata.get('category', 'general')}] {i.page_content}"
                    for i in identities
                ]
            )

            # ── Build recent conversation window (last 6 turns, excluding current) ──
            recent_msgs = st.session_state.messages[-7:-1]  # Exclude the just-added user message
            recent_messages = "\n".join(
                [f"{'Human' if m['role'] == 'user' else 'AI'}: {m['content'][:500]}" for m in recent_msgs]
            ) if recent_msgs else "(First message in this session)"

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
                        "recent_messages": recent_messages,
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
                            "recent_messages": recent_messages,
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

    # ── ORGANIC LEARNING: Save to Episodic Memory (timestamped + tagged) ──
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    # Compact summary format — easier for retrieval than raw conversation
    memory_string = f"[{timestamp}] User asked about: {user_input[:300]}\nAI response summary: {response_text[:500]}"
    chat_db.add_texts(
        [memory_string],
        metadatas=[
            {
                "timestamp": datetime.now().isoformat(),
                "type": "conversation",
                "user_message": user_input[:500],
            }
        ],
    )

    # ── PASSIVE IDENTITY EXTRACTION: Mine facts in background ──
    facts_stored = extract_personal_facts(user_input, response_text, identity_db, llm)
    if facts_stored > 0:
        st.toast(f"🧬 Learned {facts_stored} new fact(s) about you!", icon="🧠")