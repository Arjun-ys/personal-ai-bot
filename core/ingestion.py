import os
import re
import time
from datetime import datetime
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import (
    GEMINI_VISION_MODEL,
    VISION_OCR_PROMPT,
    _ATOMIC_PATTERN,
    _TABULAR_DOC_PATTERNS,
)

# Gemini Vision (for table-aware OCR) — uses modern google-genai SDK
try:
    from google import genai as google_genai
    GEMINI_VISION_AVAILABLE = True
except ImportError:
    GEMINI_VISION_AVAILABLE = False


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


# Aliases for flexible naming
extract_text_rapidocr = extract_text_rapidocr_fallback


def extract_text_from_pdf(pdf_path, force_vision=False, progress_callback=None):
    """
    3-tier PDF extraction:
    - If force_vision is True: attempts Gemini Vision first.
    - Otherwise: attempts PyMuPDF digital text first (>50 chars).
    - If digital text is empty/insufficient: falls back to Gemini Vision.
    - If Vision is unavailable or fails: falls back to RapidOCR.
    Returns (extracted_text, extraction_method).
    """
    if force_vision:
        vision_text = extract_text_gemini_vision(pdf_path, progress_callback=progress_callback)
        if vision_text and len(vision_text.strip()) > 50:
            return vision_text, "gemini_vision"

    digital_text = extract_text_digital(pdf_path)
    if digital_text and len(digital_text.strip()) > 50:
        return digital_text, "pymupdf_digital"

    vision_text = extract_text_gemini_vision(pdf_path, progress_callback=progress_callback)
    if vision_text and len(vision_text.strip()) > 50:
        return vision_text, "gemini_vision"

    fallback_text = extract_text_rapidocr_fallback(pdf_path)
    if fallback_text:
        return fallback_text, "rapidocr_fallback"

    return "", "none"


def _split_preserving_atomic_blocks(content):
    """Split content preserving markdown tables and code blocks as atomic units."""
    return re.split(_ATOMIC_PATTERN, content)


def _chunk_markdown_table(stripped, base_meta=None, current_section=""):
    """Large table? Split by row batches, preserve header."""
    base_meta = base_meta or {}
    chunk_docs = []
    if len(stripped) > 3000:
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
                metadata={**base_meta, "content_type": "table",
                          "section": current_section},
            )
        )
    return chunk_docs


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
        parts = _split_preserving_atomic_blocks(content)

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
                if is_table:
                    chunk_docs.extend(
                        _chunk_markdown_table(stripped, base_meta, current_section)
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

