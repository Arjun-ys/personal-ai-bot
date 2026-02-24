# 🧠 Personal AI Agent

A local-first RAG system that acts as a **persistent digital extension of yourself**. It doesn't just answer questions — it organically learns your coding style, tech preferences, personality, and personal context through continuous conversation.

Built with Python, Streamlit, LangChain, and ChromaDB.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-green)
![LangChain](https://img.shields.io/badge/LangChain-Orchestration-yellow)

---

## How It Works

### Triple-Memory Architecture

The system maintains **three isolated ChromaDB collections** to separate concerns and prevent data corruption:

| Collection | Purpose | Written When |
|---|---|---|
| **Document Vault** | Factual knowledge from uploaded files (PDF, TXT, LOG) | On file upload |
| **Episodic Memory** | Timestamped conversation history | After every chat turn |
| **Core Identity** | Extracted personal facts, preferences, and communication style | Mined passively after every chat |

Every time you chat, the AI:
1. Retrieves relevant context from **all three** memory banks
2. Generates a response using your chosen LLM
3. Saves the conversation to episodic memory
4. Runs a background **fact extraction pass** that mines personal details from your message and deduplicates them before storing in the identity database

Over days and weeks, the identity collection builds into a rich profile — your name, education, tech stack, projects, opinions, and even how you communicate.

---

### Three-Tier OCR Engine

Uploading scanned documents (like academic marksheets) triggers a cascading OCR pipeline:

| Tier | Engine | Speed | Table Support | When Used |
|---|---|---|---|---|
| 1 | **PyMuPDF** (digital text) | ⚡ Instant | ❌ | Native-text PDFs |
| 2 | **Gemini Vision** | 🔄 ~2s/page | ✅ Full markdown tables | Scanned/complex PDFs |
| 3 | **RapidOCR** (local) | 🐌 Slow | ❌ | Offline fallback |

**Tier 2 (Gemini Vision)** is the key innovation — it renders each page at 300 DPI, sends it to Gemini's vision model with a strict prompt that forces:
- Markdown table output with subject-score row alignment
- Watermark/border noise rejection
- Numerical precision for scores, dates, and roll numbers

### Table-Preserving Chunker

Standard text splitters destroy table structure. This system:
- Detects markdown tables via regex before splitting
- Keeps tables as **atomic chunks** (never split mid-row)
- Splits large tables by row batches with the header repeated on each chunk
- Tags every chunk with `content_type: "text"` or `"table"` metadata

---

## Features

- **Streaming responses** — tokens appear in real-time
- **Dual LLM support** — Local (Ollama/Llama 3) or Cloud (Gemini Flash)
- **Memory Dashboard** — live counts for all three collections
- **Identity Browser** — view all extracted facts about you in the sidebar
- **Memory Management** — per-collection clear buttons
- **Full Memory Audit** — ask the AI to compile everything it knows about you
- **Document deduplication** — prevents re-ingesting the same file
- **Recent conversation window** — last 6 messages fed into prompt for coherent multi-turn dialogue

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running (for local mode)
- Google API key (for Gemini cloud mode and Vision OCR)

### Installation

```bash
git clone https://github.com/Arjun-ys/personal-ai-bot.git
cd personal-ai-bot

python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

### Running

```bash
# Start Ollama (for local mode)
ollama serve
ollama pull llama3

# Launch the app
streamlit run app.py
```

---

## Project Structure

```
├── app.py              # Main application (all logic in one file)
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
├── ai_memory_db/       # ChromaDB persistent storage (auto-created)
│   ├── chroma.sqlite3
│   └── .../            # Collection data directories
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Vector Store | ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| LLM (Local) | Ollama + Llama 3 |
| LLM (Cloud) | Google Gemini 2.0 Flash |
| Vision OCR | Google Gemini Vision API |
| PDF Parsing | PyMuPDF |
| OCR Fallback | RapidOCR |
| Orchestration | LangChain |

---

## Usage Tips

1. **Start conversations naturally** — tell it your name, what you're working on, what tools you use. It extracts and remembers these automatically.
2. **Upload your documents** — resumes, marksheets, project docs, logs. They go into the vault and are never modified.
3. **Check the identity panel** — expand "🧬 View Known Facts About You" to see what it has learned.
4. **Run memory audits** — click "🔎 Full Memory Audit" to get a full dossier of everything it knows.
5. **Use local mode for privacy** — Ollama keeps everything on your machine. Switch to Gemini when you need stronger reasoning.

---

## License

MIT
