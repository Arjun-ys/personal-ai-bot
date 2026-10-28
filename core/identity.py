import re
import sys
import json
import time as _time
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

from core.config import (
    FACT_EXTRACTION_PROMPT,
    CONTRADICTION_CHECK_PROMPT,
    DOC_FACT_PROMPT,
    _STALE_FACT_DAYS,
    _STALE_CHECK_INTERVAL_TURNS,
)


def _log(msg: str):
    """Safely print log messages without UnicodeEncodeError on Windows console."""
    try:
        print(msg)
    except UnicodeEncodeError:
        try:
            encoding = sys.stdout.encoding or "utf-8"
            print(msg.encode(encoding, errors="replace").decode(encoding))
        except Exception:
            pass


# ╔══════════════════════════════════════════════════════╗
# ║  1. PYDANTIC STRUCTURED OUTPUT SCHEMAS               ║
# ╚══════════════════════════════════════════════════════╝

class ExtractedFact(BaseModel):
    fact: str
    category: Literal[
        "identity",
        "preference",
        "skill",
        "project",
        "goal",
        "relationship",
        "constraint",
        "opinion",
        "general",
    ]
    confidence: Literal["high", "medium", "low"]


class ExtractedFactList(BaseModel):
    facts: list[ExtractedFact]


class FactResolution(BaseModel):
    new_fact_index: int
    relation: Literal["updates", "contradicts", "compatible"]
    target_existing_id: Optional[str] = None
    clarification_question: Optional[str] = None


class FactResolutionList(BaseModel):
    resolutions: list[FactResolution]


class ClarificationOutcome(BaseModel):
    resolved_fact: Optional[str] = None
    category: str = "general"
    delete_existing_id: Optional[str] = None


# ╔══════════════════════════════════════════════════════╗
# ║  2. HELPER FUNCTIONS                                 ║
# ╚══════════════════════════════════════════════════════╝

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


def _add_to_clarification_queue(old_fact, new_fact, category, question=None):
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
            "question": question,
            "detected_at": datetime.now().isoformat(),
        })
    except Exception:
        pass


# ╔══════════════════════════════════════════════════════╗
# ║  3. FACT EXTRACTION & DEDUPLICATION                   ║
# ╚══════════════════════════════════════════════════════╝

def extract_personal_facts(user_input, ai_response, identity_db=None, llm=None):
    """
    Mine personal facts from a conversation turn using Pydantic structured output.
    Returns the validated list of dicts.
    """
    if llm is None:
        return []

    try:
        structured_llm = llm.with_structured_output(ExtractedFactList)
        chain = FACT_EXTRACTION_PROMPT | structured_llm
        result = chain.invoke(
            {"user_input": user_input, "ai_response": ai_response}
        )

        if result and hasattr(result, "facts") and result.facts:
            return [f.model_dump() for f in result.facts if len(f.fact.strip()) >= 5]
        elif isinstance(result, dict) and "facts" in result:
            return [
                f if isinstance(f, dict) else f.model_dump()
                for f in result["facts"]
                if (f.get("fact", "") if isinstance(f, dict) else f.fact).strip() >= 5
            ]
        elif isinstance(result, list):
            return [f if isinstance(f, dict) else f.model_dump() for f in result]
        return []
    except Exception as e:
        err_str = str(e).lower()
        if "429" in err_str or "quota" in err_str or "rate" in err_str:
            _log(f"[Identity Extraction] Rate limit hit: {e}")
            raise  # Re-raise so the retry loop in the background thread can catch it
        _log(f"[Identity Extraction Error] {e}")
        return []


def save_extracted_facts(facts, identity_db, llm=None, source_snippet=""):
    """
    Save extracted facts into identity_db using cosine-distance deduplication
    and single-pass batched LLM resolution for ambiguous candidate pairs.
    """
    if not facts or not isinstance(facts, list):
        return 0

    stored_count = 0
    ambiguous_pairs = []  # List of tuples: (new_fact_index, new_fact_dict, matching_candidates)
    direct_inserts = []

    for idx, fact_item in enumerate(facts):
        if isinstance(fact_item, BaseModel):
            fact_dict = fact_item.model_dump()
        elif isinstance(fact_item, dict):
            fact_dict = fact_item
        else:
            continue

        fact_text = fact_dict.get("fact", "").strip()
        if len(fact_text) < 5:
            continue

        category = fact_dict.get("category", "general")
        confidence = fact_dict.get("confidence", "medium")

        # Query identity_db for top matches
        results = None
        try:
            results = identity_db._collection.query(
                query_texts=[fact_text],
                n_results=3,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            pass

        if results and results.get("distances") and results["distances"][0]:
            distances = results["distances"][0]
            ids = results["ids"][0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]

            # 1. Near duplicate check: distance < 0.25 (Cosine distance)
            if distances[0] < 0.25:
                closest_id = ids[0]
                closest_meta = metas[0] if metas else {}
                try:
                    identity_db._collection.update(
                        ids=[closest_id],
                        metadatas=[{
                            **closest_meta,
                            "last_seen": datetime.now().isoformat(),
                            "confidence": confidence,
                        }],
                    )
                except Exception:
                    pass
                continue  # Near-duplicate: updated last_seen, skip further storage

            # 2. Check candidates in 0.25 <= distance < 0.65 with same category
            matching_candidates = []
            for c_idx, dist in enumerate(distances):
                cand_meta = metas[c_idx] if c_idx < len(metas) else {}
                cand_cat = cand_meta.get("category", "general")
                if 0.25 <= dist < 0.65 and cand_cat == category:
                    matching_candidates.append({
                        "id": ids[c_idx],
                        "text": docs[c_idx] if c_idx < len(docs) else "",
                        "category": cand_cat,
                        "distance": dist,
                    })

            if matching_candidates:
                ambiguous_pairs.append((idx, fact_dict, matching_candidates))
            else:
                direct_inserts.append(fact_dict)
        else:
            direct_inserts.append(fact_dict)

    # 3. Direct inserts for facts with no near-duplicates and no ambiguous candidates
    for item in direct_inserts:
        identity_db.add_texts(
            [item["fact"]],
            metadatas=[{
                "category": item.get("category", "general"),
                "confidence": item.get("confidence", "medium"),
                "extracted_at": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "source_snippet": source_snippet[:200],
            }],
        )
        stored_count += 1

    # 4. Batched resolution for ambiguous pairs
    if ambiguous_pairs:
        resolutions_map = {}
        if llm is not None:
            batch_lines = []
            for new_idx, new_f, candidates in ambiguous_pairs:
                cand_desc = "\n".join([f"    - ID: \"{c['id']}\" | Fact: \"{c['text']}\"" for c in candidates])
                batch_lines.append(
                    f"Item Index {new_idx}:\n"
                    f"  New Fact: \"{new_f['fact']}\" (Category: {new_f.get('category', 'general')})\n"
                    f"  Existing Candidate Facts:\n{cand_desc}"
                )
            batch_text = "\n\n".join(batch_lines)

            BATCH_RESOLUTION_PROMPT = ChatPromptTemplate.from_template(
                """You are a personal fact-checking and memory reconciliation system.
Analyze the following ambiguous new facts against existing candidate facts about the SAME user.

For EACH Item Index, determine the relationship between the new fact and the existing candidate(s):
- "updates": The new fact is a newer version of, or supersedes, a specific existing candidate. Set target_existing_id to that candidate's ID.
- "contradicts": The new fact directly conflicts with an existing candidate (cannot both be true). Set target_existing_id to that candidate's ID and provide a short clarification_question to ask the user.
- "compatible": The new fact can be true at the same time as existing facts (different aspects or complementary info). Set target_existing_id to null and clarification_question to null.

Ambiguous Facts to Analyze:
{batch_input}
"""
            )

            try:
                structured_resolver = llm.with_structured_output(FactResolutionList)
                response = (BATCH_RESOLUTION_PROMPT | structured_resolver).invoke({
                    "batch_input": batch_text
                })
                if response and hasattr(response, "resolutions") and response.resolutions:
                    for res in response.resolutions:
                        resolutions_map[res.new_fact_index] = res
                elif isinstance(response, dict) and "resolutions" in response:
                    for res in response["resolutions"]:
                        if isinstance(res, dict):
                            resolutions_map[res.get("new_fact_index")] = FactResolution(**res)
                        else:
                            resolutions_map[res.new_fact_index] = res
            except Exception as e:
                _log(f"[Batched Fact Resolution Error] {e}")

        # Apply resolutions
        for new_idx, new_f, candidates in ambiguous_pairs:
            res = resolutions_map.get(new_idx)
            relation = res.relation if res else "compatible"
            target_id = res.target_existing_id if res else None

            if relation == "updates":
                if target_id:
                    try:
                        identity_db._collection.delete(ids=[target_id])
                    except Exception:
                        pass
                identity_db.add_texts(
                    [new_f["fact"]],
                    metadatas=[{
                        "category": new_f.get("category", "general"),
                        "confidence": new_f.get("confidence", "medium"),
                        "extracted_at": datetime.now().isoformat(),
                        "last_seen": datetime.now().isoformat(),
                        "source_snippet": source_snippet[:200],
                    }],
                )
                stored_count += 1
                _log(f"🔄 Updated fact (replaced {target_id}): {new_f['fact']}")

            elif relation == "contradicts":
                old_text = ""
                if target_id:
                    old_text = next((c["text"] for c in candidates if c["id"] == target_id), "")
                if not old_text and candidates:
                    old_text = candidates[0]["text"]
                _add_to_clarification_queue(
                    old_fact=old_text,
                    new_fact=new_f["fact"],
                    category=new_f.get("category", "general"),
                    question=res.clarification_question if res else None,
                )
                _log(f"⚠️ Contradiction queued: \"{old_text}\" vs \"{new_f['fact']}\"")

            else:  # "compatible"
                identity_db.add_texts(
                    [new_f["fact"]],
                    metadatas=[{
                        "category": new_f.get("category", "general"),
                        "confidence": new_f.get("confidence", "medium"),
                        "extracted_at": datetime.now().isoformat(),
                        "last_seen": datetime.now().isoformat(),
                        "source_snippet": source_snippet[:200],
                    }],
                )
                stored_count += 1

    return stored_count


def extract_facts_from_document(doc_text, source_name, identity_db, llm):
    """
    Mine personal facts from an uploaded document (e.g. resume, transcript).
    Called once at ingestion time — not on every chat.
    """
    try:
        structured_extractor = llm.with_structured_output(ExtractedFactList)
        WINDOW_SIZE = 3000
        WINDOW_OVERLAP = 500
        total_stored = 0
        pos = 0

        while pos < len(doc_text):
            window = doc_text[pos:pos + WINDOW_SIZE]
            if len(window.strip()) < 50:
                break

            result = (DOC_FACT_PROMPT | structured_extractor).invoke({
                "doc_text": window,
                "source_name": source_name,
            })
            if result and hasattr(result, "facts") and result.facts:
                facts_data = [f.model_dump() for f in result.facts]
                total_stored += save_extracted_facts(
                    facts_data,
                    identity_db,
                    llm=llm,
                    source_snippet=f"Document: {source_name}",
                )
            elif isinstance(result, dict) and "facts" in result:
                facts_data = [
                    f if isinstance(f, dict) else f.model_dump()
                    for f in result["facts"]
                ]
                total_stored += save_extracted_facts(
                    facts_data,
                    identity_db,
                    llm=llm,
                    source_snippet=f"Document: {source_name}",
                )

            pos += WINDOW_SIZE - WINDOW_OVERLAP
            if pos >= WINDOW_SIZE * 5:
                break

        return total_stored
    except Exception as e:
        _log(f"[Document Fact Extraction Error] {e}")
        return 0


# ╔══════════════════════════════════════════════════════╗
# ║  4. CLARIFICATION HANDLING & STALENESS               ║
# ╚══════════════════════════════════════════════════════╝

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
            _log(f"📅 Found {stale_count} stale fact(s) queued for re-verification")
    except Exception as e:
        _log(f"[Stale Fact Check Error] {e}")


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

        question = item.get("question")

        if new.startswith("[STALE"):
            lines.append(
                f"{i}. I noted a while ago: *\"{old}\"* (category: {cat}). "
                f"Is this still accurate?"
            )
        elif question:
            lines.append(f"{i}. {question}")
        else:
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
    using llm.with_structured_output(ClarificationOutcome) to resolve contradictions.
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

    CLARIFICATION_OUTCOME_PROMPT = ChatPromptTemplate.from_template(
        """The user was asked to clarify conflicting facts about themselves.

Previous fact on file: "{old_fact}" (Existing ID: {existing_id})
New conflicting fact: "{new_fact}"
Category: "{category}"

User's clarification response:
"{user_response}"

Determine the outcome:
- If user confirms the new fact or updates their information: set resolved_fact to the confirmed/updated fact, category to "{category}", and delete_existing_id to "{existing_id}" (if provided).
- If user confirms the previous fact (rejecting the new one): set resolved_fact to the previous fact, category to "{category}", and delete_existing_id to null.
- If user rejects both or wants it removed: set resolved_fact to null, category to "{category}", and delete_existing_id to "{existing_id}".
- If both are true: set resolved_fact to the unified/appropriate fact, and delete_existing_id to null.
"""
    )

    try:
        structured_outcome_llm = llm.with_structured_output(ClarificationOutcome)

        for item in shown_items:
            existing_id = None
            try:
                old_res = identity_db._collection.query(
                    query_texts=[item["old_fact"]],
                    n_results=1,
                )
                if old_res and old_res.get("ids") and old_res["ids"][0]:
                    if old_res.get("distances") and old_res["distances"][0] and old_res["distances"][0][0] < 0.35:
                        existing_id = old_res["ids"][0][0]
            except Exception:
                existing_id = None

            try:
                outcome = (CLARIFICATION_OUTCOME_PROMPT | structured_outcome_llm).invoke({
                    "old_fact": item["old_fact"],
                    "new_fact": item["new_fact"],
                    "category": item["category"],
                    "existing_id": existing_id or "none",
                    "user_response": user_input,
                })

                if outcome:
                    if isinstance(outcome, dict):
                        resolved_fact = outcome.get("resolved_fact")
                        cat = outcome.get("category") or item["category"]
                        delete_id = outcome.get("delete_existing_id")
                    else:
                        resolved_fact = outcome.resolved_fact
                        cat = outcome.category or item["category"]
                        delete_id = outcome.delete_existing_id

                    del_id = delete_id or (
                        existing_id if resolved_fact and resolved_fact != item["old_fact"] else None
                    )
                    if del_id and del_id != "none":
                        try:
                            identity_db._collection.delete(ids=[del_id])
                        except Exception:
                            pass

                    if resolved_fact and len(resolved_fact.strip()) >= 5:
                        identity_db.add_texts(
                            [resolved_fact.strip()],
                            metadatas=[{
                                "category": cat,
                                "confidence": "high",
                                "extracted_at": datetime.now().isoformat(),
                                "last_seen": datetime.now().isoformat(),
                                "source_snippet": "User clarification",
                            }],
                        )
                        _log(f"✅ Clarification resolved: {resolved_fact}")
            except Exception as item_err:
                _log(f"[Clarification Item Error] {item_err}")

        # Remove processed items from queue
        try:
            import streamlit as st
            st.session_state["_clarification_queue"] = queue[len(shown_items):]
        except Exception:
            pass

    except Exception as e:
        _log(f"[Clarification Resolution Error] {e}")
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
            facts = extract_personal_facts(u_input, ai_resp, identity_db, extraction_llm)
            facts_stored = 0
            if facts and isinstance(facts, list):
                facts_stored = save_extracted_facts(facts, identity_db, extraction_llm, source_snippet=u_input)
            elif isinstance(facts, int):
                facts_stored = facts

            if facts_stored > 0:
                # Store result so the UI can show it on next rerun
                try:
                    import streamlit as st
                    st.session_state["_last_extraction_result"] = f"🧬 Learned {facts_stored} new fact(s) about you!"
                except Exception:
                    pass
                _log(f"🧬 Learned {facts_stored} new fact(s) about you!")
            return facts_stored
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "rate" in err_str:
                wait = (attempt + 1) * 10  # 10s, 20s, 30s — generous waits
                _log(f"[Identity Extraction] Rate limited, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                _time.sleep(wait)
            else:
                try:
                    import streamlit as st
                    st.session_state["_last_extraction_result"] = f"⚠️ Extraction error: {str(e)[:100]}"
                except Exception:
                    pass
                _log(f"[Identity Extraction Error] {e}")
                return 0
    return 0
