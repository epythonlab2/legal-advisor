"""
Pipeline Orchestrator for Ethiopian Legal RAG System

- Loads legal documents (Negarit Gazette)
- Splits documents into chunks
- Builds vector store embeddings
- Optionally starts a FastAPI RAG server
"""

import argparse
import sys
from pathlib import Path

import uvicorn

from src.embeddings.vector_builder import build_vector_store

# Modular pipeline components
from src.ingestion.loader import load_documents
from src.ingestion.splitter import split_documents
from src.utils.logger import get_logger

logger = get_logger("pipeline_orchestrator")

# -------------------------------------------------------------------
# Default Configuration
# -------------------------------------------------------------------

DATA_DIR = Path("data/raw")
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 200


# -------------------------------------------------------------------
# Pipeline Steps
# -------------------------------------------------------------------


def run_pipeline(
    data_dir: Path = DATA_DIR,
) -> None:
    """
    Run full Legal RAG pipeline:
    1. Load documents
    2. Split into chunks
    3. Build vector store embeddings
    """
    logger.info("=== Starting Legal RAG Pipeline ===")

    try:
        # Step 1: Load documents
        documents = load_documents(data_dir)
        if not documents:
            logger.warning("No documents loaded from %s. Pipeline exiting.", data_dir)
            return

        # Step 2: Split documents into chunks
        chunks = split_documents(documents)
        if not chunks:
            logger.warning("No document chunks created. Pipeline exiting.")
            return

        # Step 3: Build vector store
        build_vector_store(chunks)
        logger.info("=== Pipeline completed successfully ===")

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)


# -------------------------------------------------------------------
# API Server
# -------------------------------------------------------------------


def start_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = True) -> None:
    """
    Start the FastAPI RAG API for querying vector store
    """
    try:
        logger.info("Starting RAG API at %s:%d", host, port)
        uvicorn.run("src.api.rag_api:app", host=host, port=port, reload=reload)
    except Exception as exc:
        logger.exception("Failed to start API: %s", exc)


# -------------------------------------------------------------------
# Command-Line Interface
# -------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Legal RAG Pipeline Orchestrator for Ethiopian Law"
    )
    parser.add_argument(
        "--run_pipeline",
        action="store_true",
        help="Run the ingestion → splitting → embedding pipeline",
    )
    parser.add_argument(
        "--start_api",
        action="store_true",
        help="Start the FastAPI RAG server after building vector store",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    return parser.parse_args()


# -------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if not (args.run_pipeline or args.start_api):
        logger.error("No action specified. Use --run_pipeline or --start_api.")
        sys.exit(1)

    if args.run_pipeline:
        run_pipeline()

    if args.start_api:
        start_api(host=args.host, port=args.port)
