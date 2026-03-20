"""
Robust Loader for Ethiopian Legal Documents

Handles inconsistent formatting in Negarit Gazette PDFs by:
- Loading each page as a separate document
- Detecting language (Amharic / English)
- Optionally detecting headings (PART, section titles)
- Flexible chunking for embedding / retrieval
- Attaching rich metadata
"""

from pathlib import Path
from typing import List, Optional
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger("document_loader")


# -----------------------------
# Regex Patterns
# -----------------------------

# Detect Amharic
AMHARIC_PATTERN = re.compile(r"[\u1200-\u137F]")

# Detect headings (optional, heuristic)
HEADING_PATTERN = re.compile(
    r"^(PART\s+[A-Z]+|ክፍል\s*\d+|PART\s+\d+|[A-Z ]{5,})", re.MULTILINE
)


# -----------------------------
# Language Detection
# -----------------------------

def detect_language(text: str) -> str:
    """Detect language based on Unicode ranges."""
    if not text:
        return "unknown"
    return "amharic" if AMHARIC_PATTERN.search(text) else "english"


# -----------------------------
# Flexible Chunking
# -----------------------------

def chunk_text(
    text: str, 
    max_chars: int = 1000, 
    overlap: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks of roughly max_chars length.
    
    Args:
        text: Full text string
        max_chars: Maximum chunk length
        overlap: Overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + max_chars, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max_chars - overlap

    return chunks


# -----------------------------
# Optional Heading Detection
# -----------------------------

def detect_heading(text: str) -> Optional[str]:
    """
    Attempt to extract a heading from the text.
    Returns the first matching line or None.
    """
    match = HEADING_PATTERN.search(text)
    return match.group(1).strip() if match else None


# -----------------------------
# Main Loader
# -----------------------------

def load_documents(data_dir: Path) -> List[Document]:
    """
    Load PDFs from a directory, chunk pages, detect headings and language,
    attach rich metadata.

    Args:
        data_dir: Directory containing PDF files

    Returns:
        List of LangChain Documents
    """
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Invalid data directory: {data_dir}")

    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", data_dir)
        return []

    all_documents: List[Document] = []

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
        except Exception as exc:
            logger.exception("Failed to load PDF %s: %s", pdf_path.name, exc)
            continue

        logger.info("Loaded %s (%d pages)", pdf_path.name, len(pages))

        for page in pages:
            text = page.page_content or ""
            language = detect_language(text)
            heading = detect_heading(text)

            # Chunk the page text
            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": pdf_path.name,
                        "file_path": str(pdf_path.resolve()),
                        "document_type": "proclamation",
                        "publisher": "Negarit Gazette",
                        "language": language,
                        "page_number": page.metadata.get("page", 0),
                        "heading": heading,
                        "chunk_index": i,
                        "chunk_size": len(chunk),
                    }
                )
                all_documents.append(doc)

    logger.info("Total processed document chunks: %d", len(all_documents))
    return all_documents