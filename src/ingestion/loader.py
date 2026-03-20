import re
from pathlib import Path
from typing import List

import fitz  # pip install pymupdf
from langchain_core.documents import Document

from src.utils.logger import get_logger

logger = get_logger("document_loader")

# Regex to detect Amharic legal structure markers
LEGAL_SECTION_REGEX = re.compile(r"^(ክፍል|ምዕራፍ|አንቀጽ)\s*[:\-–]?\s*(.*)", re.MULTILINE)


def load_documents(data_dir: Path) -> List[Document]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Invalid directory: {data_dir}")

    pdf_files = sorted(data_dir.glob("*.pdf"))
    amharic_documents = []

    for pdf_path in pdf_files:
        try:
            doc = fitz.open(str(pdf_path))
            logger.info(f"Processing {pdf_path.name} with Amharic-only parsing...")

            for page_num, page in enumerate(doc):
                width, height = page.rect.width, page.rect.height
                left_col = fitz.Rect(0, 0, width / 2, height)  # Amharic column

                amharic_text = page.get_text("text", clip=left_col).strip()

                if amharic_text:
                    # Try to detect legal section (Part / Chapter / Article)
                    section_match = LEGAL_SECTION_REGEX.search(amharic_text)
                    section_type, section_title = None, None
                    if section_match:
                        section_type = section_match.group(1)
                        section_title = section_match.group(2).strip()

                    amharic_documents.append(
                        Document(
                            page_content=amharic_text,
                            metadata={
                                "source": pdf_path.name,
                                "language": "amharic",
                                "page_number": page_num + 1,
                                "column": "left",
                                "section_type": section_type,
                                "section_title": section_title,
                            },
                        )
                    )

            doc.close()
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")

    logger.info(f"Extracted {len(amharic_documents)} Amharic pages with sections")
    return amharic_documents
