# ingestion/splitter.py

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logger import get_logger

logger = get_logger("document_chunker")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 200


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks, respecting legal structure.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of document chunks.
    """

    if not documents:
        logger.warning("No documents provided for splitting")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\nArticle",
            "\n\nSection",
            "\n\n",
            "\n",
            " ",
        ],
    )

    chunks = splitter.split_documents(documents)
    logger.info("Created %d document chunks", len(chunks))

    return chunks
