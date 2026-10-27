from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from core.config import (
    PERSIST_DIR,
    EMBEDDING_MODEL,
    _BROAD_QUERY_PATTERNS,
    CONVERSATION_SUMMARY_PROMPT,
    _SUMMARIZE_EVERY_N_TURNS,
)


def init_databases(persist_dir=PERSIST_DIR, embedding_model=EMBEDDING_MODEL):
    """Initialize three isolated ChromaDB collections for the memory system."""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # Collection A: THE VAULT — Strictly for uploaded documents (PDF, TXT, LOG)
    doc_db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="document_vault",
    )

    # Collection B: EPISODIC MEMORY — Every chat turn is saved here
    chat_db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="episodic_memory",
    )

    # Collection C: CORE IDENTITY — Extracted personal facts & preferences
    # This is the "personality crystal" that grows over time.
    identity_db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name="core_identity",
    )

    return doc_db, chat_db, identity_db


get_vector_stores = init_databases


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


_detect_referenced_sources = _match_document_source


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


def _get_recent_episodic_memories(chat_db, k=12):
    """
    Retrieve the most recent episodic memories by timestamp (NO semantic filter).
    Used for meta-queries like "what did we talk about recently?"
    """
    try:
        all_data = chat_db._collection.get(
            limit=200,
            include=["documents", "metadatas"],
        )
        if not all_data or not all_data["documents"]:
            return []

        # Pair documents with timestamps, sort by recency
        entries = []
        for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
            ts_str = meta.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                ts = datetime.min
            entries.append((ts, doc, meta))

        entries.sort(key=lambda x: x[0], reverse=True)
        return [
            Document(page_content=doc, metadata=meta)
            for _, doc, meta in entries[:k]
        ]
    except Exception:
        return []


def _get_all_identity_facts(identity_db):
    """
    Pull ALL identity facts (not just top-k similarity).
    Used when the user asks the bot to recall/remember something specific.
    """
    try:
        data = identity_db._collection.get(
            limit=500,
            include=["documents", "metadatas"],
        )
        if not data or not data["documents"]:
            return []
        return [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(data["documents"], data["metadatas"])
        ]
    except Exception:
        return identity_db.similarity_search("personal facts", k=20)


def _retrieve_full_source_chunks(source_name, doc_db, limit=100):
    """Retrieve all chunks belonging to a specific document source."""
    try:
        results = doc_db._collection.get(
            where={"source": source_name},
            limit=limit,
            include=["documents", "metadatas"],
        )
        if results and results["documents"]:
            return [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(results["documents"], results["metadatas"])
            ]
    except Exception:
        pass
    return []


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
    matched_source = _detect_referenced_sources(query, sources)

    if matched_source:
        chunks = _retrieve_full_source_chunks(matched_source, doc_db, limit=100)
        if chunks:
            return chunks

    # Default: similarity search
    return doc_db.similarity_search(query, k=default_k)


def _consolidate_episodic_memory(messages, chat_db, llm, last_summary_turn=None):
    """
    If the session has accumulated N user turns since last summary,
    generate a consolidated summary and store it in episodic memory.
    """
    user_turn_count = sum(1 for m in messages if m["role"] == "user")
    if last_summary_turn is None:
        try:
            import streamlit as st
            last_summary_turn = st.session_state.get("_last_summary_turn", 0)
        except Exception:
            last_summary_turn = 0

    if user_turn_count - last_summary_turn < _SUMMARIZE_EVERY_N_TURNS:
        return last_summary_turn  # Not enough new turns yet

    # Build conversation text from turns since last summary
    recent_turns = messages[last_summary_turn * 2:]  # Rough: 2 messages per turn
    if len(recent_turns) < 4:
        return last_summary_turn

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
            try:
                import streamlit as st
                st.session_state["_last_summary_turn"] = user_turn_count
            except Exception:
                pass
            print(f"📝 Session summary saved (turns {last_summary_turn}-{user_turn_count})")
            return user_turn_count
    except Exception as e:
        print(f"[Session Summary Error] {e}")

    return last_summary_turn


_maybe_summarize_session = _consolidate_episodic_memory


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
        feedback = f"✅ {label} cleared ({total_deleted} entries removed)." if total_deleted > 0 else f"ℹ️ {label} is already empty."
        try:
            import streamlit as st
            st.session_state.clear_feedback = feedback
            st.rerun()
        except Exception:
            pass
        return total_deleted
    except Exception as e:
        try:
            import streamlit as st
            st.sidebar.error(f"Error clearing {label}: {e}")
        except Exception:
            print(f"Error clearing {label}: {e}")
        return 0

