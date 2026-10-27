import re
import json
import time as _time
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate

from core.config import (
    FACT_EXTRACTION_PROMPT,
    CONTRADICTION_CHECK_PROMPT,
    DOC_FACT_PROMPT,
    _STALE_FACT_DAYS,
    _STALE_CHECK_INTERVAL_TURNS,
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
    try:
        import streamlit as st
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
    except Exception:
        pass


def save_extracted_facts(facts, identity_db, source_snippet=""):
    """Store extracted fact dictionaries into identity_db."""
    stored = 0
    for item in facts:
        fact_text = item.get("fact", "").strip() if isinstance(item, dict) else str(item).strip()
        if len(fact_text) < 5:
            continue
        category = item.get("category", "general") if isinstance(item, dict) else "general"
        confidence = item.get("confidence", "medium") if isinstance(item, dict) else "medium"
        identity_db.add_texts(
            [fact_text],
            metadatas=[{
                "category": category,
                "confidence": confidence,
                "extracted_at": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "source_snippet": source_snippet[:200],
            }],
        )
        stored += 1
    return stored


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


def _check_stale_facts(identity_db):
    """
    Scan identity facts for staleness (not seen in 30+ days).
    Queue stale facts for re-verification with the user.
    Only runs every N turns to avoid overhead.
    """
    try:
        import streamlit as st
        messages = st.session_state.get("messages", [])
        user_turn_count = sum(1 for m in messages if m["role"] == "user")
        last_stale_check = st.session_state.get("_last_stale_check_turn", 0)

        if user_turn_count - last_stale_check < _STALE_CHECK_INTERVAL_TURNS:
            return  # Not time yet

        st.session_state["_last_stale_check_turn"] = user_turn_count
    except Exception:
        pass

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
    try:
        import streamlit as st
        queue = st.session_state.get("_clarification_queue", [])
    except Exception:
        queue = []

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
    try:
        import streamlit as st
        queue = st.session_state.get("_clarification_queue", [])
    except Exception:
        queue = []

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
        try:
            import streamlit as st
            st.session_state["_clarification_queue"] = queue[len(shown_items):]
        except Exception:
            pass

    except Exception as e:
        print(f"[Clarification Resolution Error] {e}")
        # Clear shown items from queue anyway to prevent infinite loops
        try:
            import streamlit as st
            st.session_state["_clarification_queue"] = queue[len(shown_items):]
        except Exception:
            pass


def run_background_extraction(u_input, ai_resp, identity_db, extraction_llm, is_cloud=False):
    """Run identity fact extraction in background thread with retry on rate limit."""
    # Small delay when Cloud mode to avoid RPM bursts
    if is_cloud:
        _time.sleep(3)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            facts_stored = extract_personal_facts(u_input, ai_resp, identity_db, extraction_llm)
            if facts_stored > 0:
                # Store result so the UI can show it on next rerun
                try:
                    import streamlit as st
                    st.session_state["_last_extraction_result"] = f"🧬 Learned {facts_stored} new fact(s) about you!"
                except Exception:
                    pass
                print(f"🧬 Learned {facts_stored} new fact(s) about you!")
            return facts_stored
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                wait = (attempt + 1) * 10  # 10s, 20s, 30s — generous waits
                print(f"[Identity Extraction] Rate limited, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                _time.sleep(wait)
            else:
                try:
                    import streamlit as st
                    st.session_state["_last_extraction_result"] = f"⚠️ Extraction error: {str(e)[:100]}"
                except Exception:
                    pass
                print(f"[Identity Extraction Error] {e}")
                return 0
    return 0

