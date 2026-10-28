import os
import re
import logging
from langchain_core.prompts import ChatPromptTemplate

# ── LOGGING SETUP ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
bot_logger = logging.getLogger("personal_ai")

# ── DIRECTORY & STORAGE CONSTANTS ──
DB_DIR = os.getenv("DB_DIR", "./ai_memory_db")
PERSIST_DIR = DB_DIR
VAULT_DIR = os.path.join(PERSIST_DIR, "vault")
EPISODIC_DIR = os.path.join(PERSIST_DIR, "episodic")
IDENTITY_DIR = os.path.join(PERSIST_DIR, "identity")

PAGE_TITLE = "Personal AI Agent"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GEMINI_VISION_MODEL = "gemini-2.5-flash"


# ── FACT CATEGORIES & CONFIDENCE ──
class FactCategory:
    IDENTITY = "identity"
    PREFERENCES = "preferences"
    PROJECTS = "projects"
    SKILLS = "skills"
    OPINIONS = "opinions"
    PERSONAL = "personal"
    GENERAL = "general"


CONFIDENCE_WEIGHTS = {
    "high": 1.0,
    "medium": 0.7,
    "low": 0.4,
}


# ── REGEX PATTERNS & CONSTANTS ──
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```[^\n]*\n[\s\S]*?```")
_ATOMIC_PATTERN = r'(\|[^\n]+\|\n(?:\|[\s:|-]+\|\n)(?:\|[^\n]+\|\n?)*|```[^\n]*\n[\s\S]*?```)'

_TABULAR_DOC_PATTERNS = re.compile(
    r"(marksc|marks\s*card|score\s*card|result|transcript"
    r"|grade\s*sheet|report\s*card|certificate)",
    re.IGNORECASE,
)

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

_BROAD_QUERY_PATTERNS = re.compile(
    r"\b(list.*(everything|all)|everything you know|tell me everything|what do you know"
    r"|all about me|full (audit|summary|overview|recap)|summarize|what.*have.*on me"
    r"|show me everything|what.*remember)\b",
    re.IGNORECASE,
)

_RECENCY_META_PATTERNS = re.compile(
    r"what (did|were) we (talk|chat|discuss|speak|say)|"
    r"(recently|last time|earlier|before|previous).*?(talk|chat|discuss|convers|spoke|said)|"
    r"(talk|chat|discuss|convers|spoke|said).*?(recently|last time|earlier|before|previous)|"
    r"we (spoke|talked|chatted|discussed).*?(recently|about|last)|"
    r"our (last|recent|previous) (conversation|chat|discussion)|"
    r"do you remember (what|our|the)|"
    r"what was (the|our) (last|recent|previous)",
    re.IGNORECASE,
)

_MEMORY_RECALL_PATTERNS = re.compile(
    r"i told you|you (don.?t |didn.?t |should )?(remember|forget|recall|know)|"
    r"you (missed|forgot|left out|didn.?t (list|mention|include|say))|"
    r"there.?s (another|one more|a .+ you)|i (need|want) you to remember|"
    r"you already know|we (already|just) (talked|spoke|discussed)|"
    r"i (already |just )?(mentioned|said|shared|told)|what do you know about me",
    re.IGNORECASE,
)

_DATA_KEYWORDS = [
    "resume", "project", "marks", "score", "document", "doc",
    "skill", "list", "tell me", "what do you know", "everything",
    "audit", "show me", "vault", "upload", "file", "find",
    "check", "content", "what's in", "report", "certificate",
    "markscard", "transcript", "result", "card", "read",
    "analyze", "summarize", "summary", "detail", "info",
]

_CASUAL_PATTERNS = [
    "i'm ", "i am ", "my ", "i have ", "i got ",
    "i live", "i work", "i moved", "i like", "i love",
    "i hate", "i prefer", "i need", "i want",
]

_STALE_FACT_DAYS = 30
_STALE_CHECK_INTERVAL_TURNS = 15
_SUMMARIZE_EVERY_N_TURNS = 10


# ── LLM PROMPTS ──
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

CONVERSATION_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """Summarize this conversation between a human and their AI assistant.
Focus on: key topics discussed, decisions made, facts revealed about the user,
tasks completed, and any important context for future conversations.

<conversation>
{conversation_text}
</conversation>

Write a concise summary (3-6 sentences). Start with "Session summary:" """
)

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
14. MEMORY RECALL: When the user says "you forgot", "I already told you", "you missed one", or "do you remember" — SEARCH through your context sections thoroughly before responding. NEVER ask the user to repeat something if it might be in your context. If you find it, say it. If you truly can't find it in any context section, admit it honestly and ask them to remind you.
15. When the user asks "what were we talking about recently" or "what did we discuss" — look at Past Conversations and Recent Messages to summarize recent topics. Do NOT guess or pick random topics.

CRITICAL: Never fabricate. But when you DO have relevant facts spread across context sections, synthesize them into a complete, connected answer.
{clarification_note}
Human: {user_input}
AI:"""

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
- If the user says "you forgot" or "I told you", search through ALL your context sections before responding. NEVER ask the user to repeat something — if it's in your memory, find it and say it. Only admit you don't know if it truly isn't in any context.
- When the user asks about recent conversations, look at Past Chats and Recent Chat to summarize what you discussed.
- If the user asks something and the answer truly isn't in any context, say you don't know yet.
- Skip filler words like "Sure," "I'd be happy to," or "Great question."
{clarification_note}

Human: {user_input}
AI:"""

