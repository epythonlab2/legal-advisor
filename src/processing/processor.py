from pathlib import Path
from typing import List
# import os, sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# # Get the path to the 'legal-advisor' root (two levels up from this script)
# root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# sys.path.append(root_path)


from src.utils.logger import get_logger


logger = get_logger("legal_processor")


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"


# -------------------------------------------------------------------
# Document Processing
# -------------------------------------------------------------------

def process_legal_documents(
    input_dir: Path,
    chunk_size: int = 800,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Load and split PDF legal documents into smaller chunks.

    Args:
        input_dir: Directory containing PDF documents
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of processed document chunks
    """

    if not input_dir.exists() or not input_dir.is_dir():
        logger.error("Invalid input directory: %s", input_dir)
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", input_dir)
        return []

    logger.info("Found %d PDF documents", len(pdf_files))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True
    )

    documents: List[Document] = []

    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            for page in pages:
                page.metadata["filename"] = pdf_path.name

            documents.extend(pages)

            logger.info("Loaded document: %s", pdf_path.name)

        except Exception:
            logger.exception("Failed to process document: %s", pdf_path.name)

    if not documents:
        logger.warning("No documents were successfully loaded")
        return []

    chunks = splitter.split_documents(documents)

    logger.info("Generated %d document chunks", len(chunks))

    return chunks


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = process_legal_documents(INPUT_DIR)

    if not chunks:
        logger.warning("No chunks generated")
        return

    logger.info("Processing complete. Total chunks: %d", len(chunks))
    logger.info("Sample metadata: %s", chunks[0].metadata)


if __name__ == "__main__":
    main()
