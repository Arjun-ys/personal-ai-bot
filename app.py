import streamlit as st
import os
import re
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import tempfile
import threading
import fitz  # PyMuPDF
from streamlit.runtime.scriptrunner import add_script_run_ctx

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

# DuckDuckGo Search (free web access — no API key needed)
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

# URL fetching
try:
    import urllib.request
    from html.parser import HTMLParser
    URL_FETCH_AVAILABLE = True
except ImportError:
    URL_FETCH_AVAILABLE = False

load_dotenv()

# ╔══════════════════════════════════════════════════════╗
# ║              CONFIGURATION                           ║
# ╚══════════════════════════════════════════════════════╝
PAGE_TITLE = "Personal AI Agent"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PERSIST_DIR = "./ai_memory_db"
GEMINI_VISION_MODEL = "gemini-2.5-flash"

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
    Split documents while keeping Markdown tables AND code blocks as atomic chunks.
    Tables/code are NEVER split mid-block. Each chunk carries its nearest section heading.
    """
    all_chunks = []

    for doc in documents:
        content = doc.page_content
        base_meta = doc.metadata.copy()

        # Regex: match markdown tables OR fenced code blocks as atomic units
        # Tables: | col | col |\n|---|---|\n| data |...
        # Code:   ```...```
        atomic_pattern = r'(\|[^\n]+\|\n(?:\|[\s:|-]+\|\n)(?:\|[^\n]+\|\n?)*|```[^\n]*\n[\s\S]*?```)'
        parts = re.split(atomic_pattern, content)

        chunk_docs = []
        text_buf = ""
        current_section = ""  # Track nearest heading for context

        for part in parts:
            stripped = part.strip()
            is_table = bool(re.match(r'\s*\|[^\n]+\|', stripped))
            is_code = stripped.startswith('```')

            # Track section headings from text parts
            if not is_table and not is_code:
                # Find the last heading in this text segment
                headings = re.findall(r'^(#{1,4}\s+.+)$', part, re.MULTILINE)
                if headings:
                    current_section = headings[-1].strip().lstrip('#').strip()

            if is_table or is_code:
                content_type = "table" if is_table else "code"

                # Flush accumulated plain text first
                if text_buf.strip():
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200
                    )
                    for tc in splitter.split_text(text_buf):
                        chunk_docs.append(
                            Document(
                                page_content=tc,
                                metadata={**base_meta, "content_type": "text",
                                          "section": current_section},
                            )
                        )
                    text_buf = ""

                # Large table? Split by row batches, preserve header
                if is_table and len(stripped) > 3000:
                    lines = stripped.split("\n")
                    header_lines = lines[:2]
                    data_lines = lines[2:]
                    header = "\n".join(header_lines)
                    batch_size = 20
                    for i in range(0, len(data_lines), batch_size):
                        batch = "\n".join(data_lines[i : i + batch_size])
                        chunk_docs.append(
                            Document(
                                page_content=f"{header}\n{batch}",
                                metadata={**base_meta, "content_type": "table",
                                          "section": current_section},
                            )
                        )
                else:
                    chunk_docs.append(
                        Document(
                            page_content=stripped,
                            metadata={**base_meta, "content_type": content_type,
                                      "section": current_section},
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
                        metadata={**base_meta, "content_type": "text",
                                  "section": current_section},
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
- Assign a confidence level to each fact:
  "high"   = directly and explicitly stated by the user (e.g. "My name is Arjun")
  "medium" = strongly implied or contextually clear (e.g. user discusses Python projects → knows Python)
  "low"    = weakly implied or ambiguous (e.g. "I might try React sometime")

Return ONLY a valid JSON array (no markdown fences, no explanation):
[{{"category": "...", "fact": "...", "confidence": "high|medium|low"}}]

If nothing found:
[]"""
)


CONTRADICTION_CHECK_PROMPT = ChatPromptTemplate.from_template(
    """You are a fact-checking system. Determine if these two facts about the SAME person contradict each other.

Existing fact: "{existing_fact}"
New fact: "{new_fact}"

Answer with ONLY one of these words:
- "contradicts" — they directly conflict (e.g., "studies CS" vs "studies AIML")
- "updates" — the new fact is a newer version of the same info (e.g., "learning Python" → "proficient in Python")
- "compatible" — they can both be true simultaneously

Answer:"""
)


def _detect_contradiction(existing_fact, new_fact, llm):
    """
    Use the LLM to determine if two facts contradict, update, or are compatible.
    Returns: "contradicts", "updates", or "compatible"
    """
    try:
        chain = CONTRADICTION_CHECK_PROMPT | llm
        result = chain.invoke({
            "existing_fact": existing_fact,
            "new_fact": new_fact,
        })
        answer = result.content.strip().lower()
        for label in ["contradicts", "updates", "compatible"]:
            if label in answer:
                return label
        return "compatible"  # Default to compatible if unclear
    except Exception:
        return "compatible"


def _add_to_clarification_queue(old_fact, new_fact, category):
    """
    Queue a contradiction for user clarification.
    The bot will ask the user to resolve it in the next conversation turn.
    """
    if "_clarification_queue" not in st.session_state:
        st.session_state["_clarification_queue"] = []

    # Avoid duplicate entries in the queue
    for item in st.session_state["_clarification_queue"]:
        if item["new_fact"] == new_fact:
            return

    st.session_state["_clarification_queue"].append({
        "old_fact": old_fact,
        "new_fact": new_fact,
        "category": category,
        "detected_at": datetime.now().isoformat(),
    })


def extract_personal_facts(user_input, ai_response, identity_db, llm):
    """
    Mine personal facts from a conversation turn with:
    - Confidence scoring (high/medium/low)
    - Contradiction detection against existing facts
    - Clarification queue for conflicts
    - Semantic dedup via ChromaDB vector distance
    """
    try:
        chain = FACT_EXTRACTION_PROMPT | llm
        result = chain.invoke(
            {"user_input": user_input, "ai_response": ai_response}
        )

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

            confidence = fact.get("confidence", "medium")
            if confidence not in ("high", "medium", "low"):
                confidence = "medium"

            category = fact.get("category", "general")

            # --- SEMANTIC DEDUPLICATION + CONTRADICTION DETECTION ---
            try:
                results = identity_db._collection.query(
                    query_texts=[fact_text],
                    n_results=3,
                    include=["documents", "metadatas", "distances"],
                )

                skip_store = False
                if results and results["distances"] and results["distances"][0]:
                    for idx, distance in enumerate(results["distances"][0]):
                        closest_id = results["ids"][0][idx]
                        closest_text = results["documents"][0][idx] if results.get("documents") else ""
                        closest_meta = results["metadatas"][0][idx] if results.get("metadatas") else {}

                        if distance < 0.4:
                            # Near-identical — update last_seen and skip
                            try:
                                identity_db._collection.update(
                                    ids=[closest_id],
                                    metadatas=[{**closest_meta,
                                                "last_seen": datetime.now().isoformat(),
                                                "confidence": confidence}],
                                )
                            except Exception:
                                pass
                            skip_store = True
                            break

                        elif distance < 0.8 and closest_meta.get("category") == category:
                            # Same category, similar topic — check for contradiction
                            verdict = _detect_contradiction(closest_text, fact_text, llm)

                            if verdict == "contradicts":
                                # Queue for user clarification instead of silently replacing
                                _add_to_clarification_queue(closest_text, fact_text, category)
                                skip_store = True
                                break
                            elif verdict == "updates":
                                # Natural evolution — replace old with new
                                identity_db._collection.delete(ids=[closest_id])
                                # Don't break — store the new version below
                            # "compatible" — store alongside

                if skip_store:
                    continue

            except Exception:
                pass

            identity_db.add_texts(
                [fact_text],
                metadatas=[
                    {
                        "category": category,
                        "confidence": confidence,
                        "extracted_at": datetime.now().isoformat(),
                        "last_seen": datetime.now().isoformat(),
                        "source_snippet": user_input[:200],
                    }
                ],
            )
            stored += 1

        return stored
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "quota" in err_str or "rate" in err_str:
            print(f"[Identity Extraction] Rate limit hit: {e}")
            raise  # Re-raise so the retry loop in the background thread can catch it
        print(f"[Identity Extraction Error] {e}")
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
        # Process document in 3000-char windows to cover full content
        WINDOW_SIZE = 3000
        WINDOW_OVERLAP = 500
        total_stored = 0
        pos = 0

        while pos < len(doc_text):
            window = doc_text[pos:pos + WINDOW_SIZE]
            if len(window.strip()) < 50:
                break  # Skip near-empty trailing window

            chain = DOC_FACT_PROMPT | llm
            result = chain.invoke({"doc_text": window, "source_name": source_name})

            raw = result.content.strip()
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if json_match:
                facts = json.loads(json_match.group())
                if facts and isinstance(facts, list):
                    for fact in facts:
                        if not isinstance(fact, dict) or "fact" not in fact:
                            continue
                        fact_text = fact["fact"].strip()
                        if len(fact_text) < 5:
                            continue

                        # Dedup via ChromaDB vector distance
                        try:
                            results = identity_db._collection.query(
                                query_texts=[fact_text],
                                n_results=1
                            )
                            if results and results["distances"] and results["distances"][0]:
                                if results["distances"][0][0] < 0.4:
                                    continue  # Near-identical — skip
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
                        total_stored += 1

            pos += WINDOW_SIZE - WINDOW_OVERLAP
            # Safety cap: max 5 windows per document to respect API limits
            if pos >= WINDOW_SIZE * 5:
                break

        return total_stored
    except Exception as e:
        print(f"[Document Fact Extraction Error] {e}")
        return 0


# ── CONVERSATION SUMMARIZATION ──
# Every N turns, consolidate the session into a high-quality summary
# stored in episodic memory. This prevents fragmentation over time.

CONVERSATION_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """Summarize this conversation between a human and their AI assistant.
Focus on: key topics discussed, decisions made, facts revealed about the user,
tasks completed, and any important context for future conversations.

<conversation>
{conversation_text}
</conversation>

Write a concise summary (3-6 sentences). Start with "Session summary:" """
)

_SUMMARIZE_EVERY_N_TURNS = 10  # Summarize every 10 user messages


def _maybe_summarize_session(messages, chat_db, llm):
    """
    If the session has accumulated N user turns since last summary,
    generate a consolidated summary and store it in episodic memory.
    """
    user_turn_count = sum(1 for m in messages if m["role"] == "user")
    last_summary_turn = st.session_state.get("_last_summary_turn", 0)

    if user_turn_count - last_summary_turn < _SUMMARIZE_EVERY_N_TURNS:
        return  # Not enough new turns yet

    # Build conversation text from turns since last summary
    recent_turns = messages[last_summary_turn * 2:]  # Rough: 2 messages per turn
    if len(recent_turns) < 4:
        return

    conv_text = "\n".join(
        [f"{'Human' if m['role'] == 'user' else 'AI'}: {m['content'][:600]}"
         for m in recent_turns[-20:]]  # Cap at last 20 messages
    )

    try:
        chain = CONVERSATION_SUMMARY_PROMPT | llm
        result = chain.invoke({"conversation_text": conv_text})
        summary = result.content.strip()

        if summary and len(summary) > 20:
            chat_db.add_texts(
                [summary],
                metadatas=[{
                    "timestamp": datetime.now().isoformat(),
                    "type": "session_summary",
                    "turn_range": f"{last_summary_turn}-{user_turn_count}",
                }],
            )
            st.session_state["_last_summary_turn"] = user_turn_count
            print(f"📝 Session summary saved (turns {last_summary_turn}-{user_turn_count})")
    except Exception as e:
        print(f"[Session Summary Error] {e}")


# ── FACT STALENESS CHECKER ──
# Flags facts that haven't been re-confirmed in a long time.
# These get added to the clarification queue for periodic re-verification.

_STALE_FACT_DAYS = 30  # Facts unseen for 30+ days are considered stale
_STALE_CHECK_INTERVAL_TURNS = 15  # Check staleness every N user turns


def _check_stale_facts(identity_db):
    """
    Scan identity facts for staleness (not seen in 30+ days).
    Queue stale facts for re-verification with the user.
    Only runs every N turns to avoid overhead.
    """
    messages = st.session_state.get("messages", [])
    user_turn_count = sum(1 for m in messages if m["role"] == "user")
    last_stale_check = st.session_state.get("_last_stale_check_turn", 0)

    if user_turn_count - last_stale_check < _STALE_CHECK_INTERVAL_TURNS:
        return  # Not time yet

    st.session_state["_last_stale_check_turn"] = user_turn_count

    try:
        all_facts = identity_db._collection.get(
            limit=200, include=["documents", "metadatas"]
        )
        if not all_facts or not all_facts["documents"]:
            return

        now = datetime.now()
        stale_count = 0

        for doc_text, meta in zip(all_facts["documents"], all_facts["metadatas"]):
            last_seen_str = meta.get("last_seen") or meta.get("extracted_at", "")
            if not last_seen_str:
                continue

            try:
                last_seen = datetime.fromisoformat(last_seen_str)
                age_days = (now - last_seen).days

                if age_days >= _STALE_FACT_DAYS:
                    confidence = meta.get("confidence", "medium")
                    category = meta.get("category", "general")

                    # Only flag low/medium confidence stale facts
                    # High-confidence facts from documents are more durable
                    if confidence != "high":
                        _add_to_clarification_queue(
                            old_fact=doc_text,
                            new_fact=f"[STALE — last confirmed {age_days} days ago] {doc_text}",
                            category=category,
                        )
                        stale_count += 1

                        # Cap at 3 stale fact checks per cycle to avoid overwhelming the user
                        if stale_count >= 3:
                            break
            except (ValueError, TypeError):
                continue

        if stale_count > 0:
            print(f"📅 Found {stale_count} stale fact(s) queued for re-verification")
    except Exception as e:
        print(f"[Stale Fact Check Error] {e}")


def _build_clarification_prompt():
    """
    Build a clarification request string from the pending queue.
    Returns None if no clarifications needed.
    """
    queue = st.session_state.get("_clarification_queue", [])
    if not queue:
        return None

    # Only surface up to 2 clarifications per turn to avoid overwhelming the user
    items = queue[:2]

    lines = ["\n\n---\n🤔 **I have a question about something I learned about you:**\n"]
    for i, item in enumerate(items, 1):
        old = item["old_fact"]
        new = item["new_fact"]
        cat = item["category"]

        if new.startswith("[STALE"):
            # Staleness re-verification
            lines.append(
                f"{i}. I noted a while ago: *\"{old}\"* (category: {cat}). "
                f"Is this still accurate?"
            )
        else:
            # Contradiction
            lines.append(
                f"{i}. I previously knew: *\"{old}\"*\n"
                f"   But now I'm seeing: *\"{new}\"* (category: {cat})\n"
                f"   Which one is correct?"
            )

    lines.append(
        "\n*You can answer naturally — I'll update my memory based on your response.*"
    )
    return "\n".join(lines)


def _process_clarification_response(user_input, identity_db, llm):
    """
    After the user responds to a clarification prompt, process their answer
    to resolve contradictions in the identity store.
    """
    queue = st.session_state.get("_clarification_queue", [])
    if not queue:
        return

    # Take the items that were shown (up to 2)
    shown_items = queue[:2]

    RESOLVE_PROMPT = ChatPromptTemplate.from_template(
        """The user was asked to clarify conflicting facts about themselves.

Shown contradictions:
{contradictions}

User's response: "{user_response}"

For EACH contradiction, determine what the user confirmed. Return a JSON array:
[{{"index": 1, "resolved_fact": "the correct fact based on user's answer", "action": "keep_old|keep_new|keep_both|discard_both"}}]

If you can't determine the answer for an item, use action "keep_both".
Return ONLY the JSON array:"""
    )

    contradictions_text = ""
    for i, item in enumerate(shown_items, 1):
        contradictions_text += f"{i}. Old: \"{item['old_fact']}\" vs New: \"{item['new_fact']}\" (category: {item['category']})\n"

    try:
        chain = RESOLVE_PROMPT | llm
        result = chain.invoke({
            "contradictions": contradictions_text,
            "user_response": user_input,
        })

        raw = result.content.strip()
        json_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not json_match:
            return

        resolutions = json.loads(json_match.group())
        if not isinstance(resolutions, list):
            return

        for res in resolutions:
            if not isinstance(res, dict):
                continue

            idx = res.get("index", 0) - 1
            if idx < 0 or idx >= len(shown_items):
                continue

            item = shown_items[idx]
            action = res.get("action", "keep_both")

            if action == "keep_new":
                # Delete old fact, store the resolved version
                try:
                    old_results = identity_db._collection.query(
                        query_texts=[item["old_fact"]], n_results=1
                    )
                    if old_results and old_results["ids"] and old_results["ids"][0]:
                        if old_results["distances"][0][0] < 0.4:
                            identity_db._collection.delete(ids=[old_results["ids"][0]])
                except Exception:
                    pass

                resolved_fact = res.get("resolved_fact", item["new_fact"])
                identity_db.add_texts(
                    [resolved_fact],
                    metadatas=[{
                        "category": item["category"],
                        "confidence": "high",  # User-confirmed = high confidence
                        "extracted_at": datetime.now().isoformat(),
                        "last_seen": datetime.now().isoformat(),
                        "source_snippet": "User clarification",
                    }],
                )
                print(f"✅ Resolved contradiction → kept new: {resolved_fact}")

            elif action == "keep_old":
                # Just refresh the old fact's last_seen
                try:
                    old_results = identity_db._collection.query(
                        query_texts=[item["old_fact"]], n_results=1
                    )
                    if old_results and old_results["ids"] and old_results["ids"][0]:
                        old_meta = old_results["metadatas"][0][0] if old_results.get("metadatas") else {}
                        identity_db._collection.update(
                            ids=[old_results["ids"][0][0]],
                            metadatas=[{**old_meta,
                                        "last_seen": datetime.now().isoformat(),
                                        "confidence": "high"}],
                        )
                except Exception:
                    pass
                print(f"✅ Resolved contradiction → kept old: {item['old_fact']}")

            elif action == "discard_both":
                try:
                    old_results = identity_db._collection.query(
                        query_texts=[item["old_fact"]], n_results=1
                    )
                    if old_results and old_results["ids"] and old_results["ids"][0]:
                        if old_results["distances"][0][0] < 0.4:
                            identity_db._collection.delete(ids=[old_results["ids"][0]])
                except Exception:
                    pass
                print(f"🗑️ Discarded both versions of contradicting fact")

            # "keep_both" → do nothing, both remain

        # Remove processed items from queue
        st.session_state["_clarification_queue"] = queue[len(shown_items):]

    except Exception as e:
        print(f"[Clarification Resolution Error] {e}")
        # Clear shown items from queue anyway to prevent infinite loops
        st.session_state["_clarification_queue"] = queue[len(shown_items):]


# ╔══════════════════════════════════════════════════════╗
# ║  4c. WEB SEARCH ENGINE                               ║
# ║      DuckDuckGo search + URL fetcher for real-time   ║
# ║      knowledge when the user asks about the world.   ║
# ╚══════════════════════════════════════════════════════╝

_WEB_QUERY_PATTERNS = re.compile(
    r"\b(what is|what are|what was|what were|who is|who are|who was|"
    r"when did|when was|when is|where is|where are|where was|"
    r"how to|how do|how does|how did|how much|how many|"
    r"latest|news|current|today|yesterday|recently|"
    r"define|explain|meaning of|tell me about|search for|look up|find out|"
    r"price of|cost of|weather|score|result|update|release date|launched|"
    r"compare|vs|versus|difference between|best|top \d+|trending|popular|"
    r"recipe|calories|distance|convert|translate|capital of|president of|"
    r"ceo of|founder of|population of|height of|age of|"
    r"controversy|feud|scandal|drama|lawsuit|sued|allegations?)\b",
    re.IGNORECASE,
)

# Explicit phrases that ALWAYS trigger web search (no pattern needed)
_EXPLICIT_WEB_PHRASES = [
    "search the web", "search online", "search the internet",
    "look up online", "look it up", "google it", "google this",
    "web search", "internet search", "find online",
    "search about", "search the web for", "look online",
    "check online", "check the internet", "check the web",
    "on the internet", "on the web", "from the web",
]

_PERSONAL_QUERY_KEYWORDS = [
    "my resume", "my project", "my skill", "my experience",
    "my education", "my certificate", "my gpa", "my cgpa",
    "my marks", "my grade", "my internship", "my achievement",
    "about me", "my name", "my age", "my college", "my university",
    "my hobby", "my interest", "my goal", "my phone", "my email",
    "my address", "my work", "my job", "my company",
]


def _is_web_query(query: str) -> bool:
    """Decide whether a user query should trigger a web search."""
    q_lower = query.lower().strip()
    # Block personal / vault queries from going to web
    for kw in _PERSONAL_QUERY_KEYWORDS:
        if kw in q_lower:
            return False
    # Explicit web-intent phrases → always trigger
    for phrase in _EXPLICIT_WEB_PHRASES:
        if phrase in q_lower:
            return True
    # Match a web-style pattern
    return bool(_WEB_QUERY_PATTERNS.search(q_lower))


def _search_web(query: str, max_results: int = 5) -> str:
    """Run a DuckDuckGo text search and return formatted results.
    Also stores raw results in session state for follow-up queries."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        # Store raw results for follow-up references ("go over the 2nd result")
        st.session_state["_last_web_results"] = results
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"[{i}] {title}\n    {body}\n    Source: {href}")
        return "\n\n".join(lines)
    except Exception as e:
        print(f"[Web Search Error] {e}")
        return ""


# ── Follow-up web result detection ──
_WEB_FOLLOWUP_PATTERN = re.compile(
    r"(?:go\s+(?:over|through|into|to)|open|read|show|expand|tell\s+me\s+(?:more\s+)?about|"
    r"explain|summarize|detail|click|fetch|visit|check|look\s+at|elaborate\s+on)"
    r".*?(?:(?:the\s+)?(?:(?:1st|first|2nd|second|3rd|third|4th|fourth|5th|fifth)\s*"
    r"(?:search\s+)?(?:result|link|article|source|one))|(?:result|link|article|source|#)\s*"
    r"(?:number\s*)?[#]?\s*([1-5]))",
    re.IGNORECASE,
)

_WEB_FOLLOWUP_NUMBER = re.compile(
    r"(?:1st|first|2nd|second|3rd|third|4th|fourth|5th|fifth|#?\s*[1-5])",
    re.IGNORECASE,
)

_ORDINAL_MAP = {
    "1st": 1, "first": 1, "1": 1,
    "2nd": 2, "second": 2, "2": 2,
    "3rd": 3, "third": 3, "3": 3,
    "4th": 4, "fourth": 4, "4": 4,
    "5th": 5, "fifth": 5, "5": 5,
}


def _check_web_followup(query: str) -> str | None:
    """If the user references a previous web result by number, fetch its full content.
    Returns formatted web_context string, or None if not a follow-up."""
    prev = st.session_state.get("_last_web_results")
    if not prev:
        return None

    match = _WEB_FOLLOWUP_PATTERN.search(query)
    if not match:
        return None

    # Extract the ordinal/number
    num_match = _WEB_FOLLOWUP_NUMBER.findall(query)
    if not num_match:
        return None

    # Take the last number found (most likely the intended one)
    raw = num_match[-1].strip().lstrip("#").strip().lower()
    idx = _ORDINAL_MAP.get(raw)
    if idx is None:
        return None

    if idx < 1 or idx > len(prev):
        return None

    result = prev[idx - 1]
    title = result.get("title", "")
    href = result.get("href", "")
    body = result.get("body", "")

    # Fetch the full article
    full_text = ""
    if href:
        full_text = _fetch_url_text(href, max_chars=10000)

    if full_text:
        return (
            f"[Full article from web result #{idx}]\n"
            f"Title: {title}\n"
            f"URL: {href}\n\n"
            f"{full_text}"
        )
    else:
        # Fallback to the snippet
        return (
            f"[Web result #{idx} — could not fetch full page]\n"
            f"Title: {title}\n"
            f"URL: {href}\n"
            f"Summary: {body}"
        )


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML→text converter for URL fetching."""

    def __init__(self):
        super().__init__()
        self._pieces: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            text = data.strip()
            if text:
                self._pieces.append(text)

    def get_text(self) -> str:
        return " ".join(self._pieces)


def _fetch_url_text(url: str, max_chars: int = 8000) -> str:
    """Download a URL and return cleaned plain text."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (PersonalAI Bot)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        extractor = _HTMLTextExtractor()
        extractor.feed(raw)
        text = extractor.get_text()
        return text[:max_chars] if len(text) > max_chars else text
    except Exception as e:
        print(f"[URL Fetch Error] {e}")
        return ""


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


def _recency_boosted_episodic_search(query, chat_db, k=8):
    """
    Fetch episodic memories with recency boost.
    1. Pull more results than needed (2x k) via similarity search.
    2. Sort by timestamp descending — recent memories float to top.
    3. Return top k results.
    This ensures the AI remembers recent conversations better.
    """
    try:
        candidates = chat_db.similarity_search(query, k=k * 2)
        if not candidates:
            return []

        # Sort by timestamp (newest first), fall back to original order
        def _get_ts(doc):
            ts = doc.metadata.get("timestamp", "")
            try:
                return datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                return datetime.min

        candidates.sort(key=_get_ts, reverse=True)
        return candidates[:k]
    except Exception:
        return chat_db.similarity_search(query, k=k)


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
                    _TABULAR_DOC_PATTERNS = re.compile(
                        r"(marksc|marks\s*card|score\s*card|result|transcript"
                        r"|grade\s*sheet|report\s*card|certificate)",
                        re.IGNORECASE,
                    )
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
# ║  6. MODEL SELECTION                                  ║
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

# ── Dedicated extraction LLM: always Gemini Flash for structured JSON tasks ──
# Fact extraction & document mining need reliable JSON output.
# Chat LLM handles the conversation (local = private & free).
# Extraction runs in background threads — API latency is invisible.
_api_key = os.getenv("GOOGLE_API_KEY")
if _api_key:
    extraction_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
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
# ║  7. TRIPLE-MEMORY UNIFIED PROMPT                     ║
# ╚══════════════════════════════════════════════════════╝

# --- CLOUD PROMPT (Gemini Flash handles complex negative rules well) ---
_cloud_prompt_text = """You are the user's personal AI — a warm, friendly second brain that knows them and grows over time.
You are NOT a corporate chatbot. You are like a trusted companion who genuinely cares.

Current date & time: {current_datetime}
You KNOW the current date and time. When the user asks what time it is, what day it is, or the date, use the value above. Do NOT say you can't tell the time.

══ CONTEXT ══
About them:
{identity_context}
Their documents:
{doc_context}
Past conversations:
{chat_context}
Web search results:
{web_context}
Recent messages:
{recent_messages}

══ RULES ══
1. Use ONLY information from the context above or from what the user says in this message.
2. If the documents section says "(No documents relevant to this message)", don't reference documents.
3. NEVER fabricate data — no guessing names, scores, dates, or facts.
4. For casual messages (hi, hey), respond naturally and warmly.
5. Only bring up background context when the user asks about it.
6. Be warm and conversational but skip filler like "Great question!" or "Let me know if you need anything."
7. When the user asks about their data, list ALL items from context completely.
8. Use markdown tables for tabular data, code blocks for code.
9. If a ══ CLARIFICATION REQUEST ══ appears below, include that question naturally at the END of your response.
10. When web search results are provided, use them to answer real-world questions. Cite sources.
11. TELLING vs ASKING:
    - If the user SHARES a fact ("My friend lives in Adelaide", "I got a job", "Her name is Nyra"), respond like a friend would — show interest, ask a natural follow-up if appropriate, and confirm you'll remember it. NEVER just say "Noted." or give a one-word response.
    - If the user ASKS a question you can't answer from context, say so honestly.
12. CONNECT THE DOTS: When the user asks about a person, topic, or relationship, look through ALL context sections (identity facts, past conversations, recent messages) to piece together everything you know. Combine related facts into a complete answer.
13. When the user gives you an instruction about your behavior ("be friendly", "remember this"), acknowledge it naturally and adjust. Don't just say "Noted."

CRITICAL: Never fabricate. But when you DO have relevant facts spread across context sections, synthesize them into a complete, connected answer.
{clarification_note}
Human: {user_input}
AI:"""

# --- LOCAL PROMPT (Llama 3 / Gemma 2 need simple, positive directives) ---
_local_prompt_text = """You are the user's personal AI — a friendly, warm companion who remembers everything about them.

Current date & time: {current_datetime}
You KNOW the current date and time. If the user asks for the time, date, or day, tell them using the value above. You CAN tell the time.

══ CONTEXT ══
Identity Facts: {identity_context}
Document Data: {doc_context}
Past Chats: {chat_context}
Web Search: {web_context}
Recent Chat: {recent_messages}

INSTRUCTIONS:
- Answer using the provided context and what the user says in this message.
- Be warm, friendly, and conversational — like a good friend, not a corporate bot.
- When the user TELLS you something new, respond with genuine interest. Ask a follow-up question or comment on what they shared. NEVER just say "Noted." or give a one-word acknowledgment.
- When the user ASKS a question, combine ALL related facts from every context section to give a complete answer.
- If the user asks something and the answer truly isn't in any context, say you don't know yet.
- Skip filler words like "Sure," "I'd be happy to," or "Great question."
{clarification_note}

Human: {user_input}
AI:"""

# Dynamically route the prompt based on the sidebar toggle
if "Local" in llm_choice:
    prompt_template = ChatPromptTemplate.from_template(_local_prompt_text)
else:
    prompt_template = ChatPromptTemplate.from_template(_cloud_prompt_text)


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

    # ── Process any pending clarification the user might be answering ──
    if st.session_state.get("_awaiting_clarification"):
        _process_clarification_response(user_input, identity_db, extraction_llm)
        st.session_state["_awaiting_clarification"] = False

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # ── Retrieve from all three memory banks ──
            # Source-aware: if user asks about "my resume", gets ALL resume chunks
            docs = retrieve_from_vault(user_input, doc_db)
            chats = _recency_boosted_episodic_search(user_input, chat_db, k=8)
            identities = identity_db.similarity_search(user_input, k=10)

            # Detect casual/short messages — don't pollute with irrelevant context
            _lower = user_input.lower()
            _data_keywords = [
                "resume", "project", "marks", "score", "document", "doc",
                "skill", "list", "tell me", "what do you know", "everything",
                "audit", "show me", "vault", "upload", "file", "find",
                "check", "content", "what's in", "report", "certificate",
                "markscard", "transcript", "result", "card", "read",
                "analyze", "summarize", "summary", "detail", "info",
            ]
            _is_casual = (
                len(user_input.split()) <= 3
                and not any(kw in _lower for kw in _data_keywords)
                and not any(p in _lower for p in [
                    "i'm ", "i am ", "my ", "i have ", "i got ",
                    "i live", "i work", "i moved", "i like", "i love",
                    "i hate", "i prefer", "i need", "i want",
                ])
            )

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
                _asks_about_docs = any(kw in _lower for kw in _data_keywords)
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
    def run_background_extraction(u_input, ai_resp):
        import time as _time
        # If both chat and extraction use Gemini (Cloud mode), wait to avoid rate limits
        if "Cloud" in llm_choice:
            _time.sleep(3)  # Space out API calls for free-tier rate limits

        max_retries = 3
        for attempt in range(max_retries):
            try:
                facts_stored = extract_personal_facts(u_input, ai_resp, identity_db, extraction_llm)
                if facts_stored > 0:
                    print(f"🧬 Learned {facts_stored} new fact(s) about you!")
                return
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "rate" in err_str:
                    wait = (attempt + 1) * 5  # 5s, 10s, 15s
                    print(f"[Identity Extraction] Rate limited, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                    _time.sleep(wait)
                else:
                    print(f"[Identity Extraction Error] {e}")
                    return

    bg_thread = threading.Thread(
        target=run_background_extraction, args=(user_input, response_text)
    )
    add_script_run_ctx(bg_thread)
    bg_thread.start()

    # ── AUTO-SUMMARIZE SESSION (periodic consolidation) ──
    _maybe_summarize_session(st.session_state.messages, chat_db, extraction_llm)

    # ── PERIODIC STALENESS CHECK (flags old facts for re-verification) ──
    _check_stale_facts(identity_db)