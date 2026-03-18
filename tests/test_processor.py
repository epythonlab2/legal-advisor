from pathlib import Path
from src.processing.processor import process_legal_documents


def test_pdf_processing(tmp_path: Path):
     # Create temporary directory
    test_dir = tmp_path / "raw"
    test_dir.mkdir()

    chunks = process_legal_documents(test_dir)

    assert isinstance(chunks, list)
    assert len(chunks) == 0