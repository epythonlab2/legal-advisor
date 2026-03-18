from pathlib import Path
from src.processing.processor import process_legal_documents


def test_pdf_processing():
    test_data = Path("data/raw")

    chunks = process_legal_documents(test_data)

    assert isinstance(chunks, list)