# ingestion/splitter_amharic.py

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logger import get_logger

logger = get_logger("document_chunker")

# Legal documents usually need slightly larger chunks to keep context together
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def split_documents(documents: List[Document]) -> List[Document]:
    if not documents:
        logger.warning("No documents provided for splitting")
        return []

    # Custom separators for Ethiopian legal texts (Amharic only)
    separators = [
        "\nአንቀጽ",  # Article
        "\nክፍል",  # Part
        "\nምዕራፍ",  # Chapter
        "\n\n",  # Paragraph break
        "\n",  # Line break
        "።",  # Amharic full stop
        " ",  # Space
        "",  # Character level fallback
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    # Clean up whitespace for each chunk
    for chunk in chunks:
        chunk.page_content = chunk.page_content.strip()

    logger.info("Created %d Amharic document chunks", len(chunks))

    return chunks
