# api/rag_api.py

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag.retriever import LegalRetriever
from src.utils.logger import get_logger

logger = get_logger("rag_api")

app = FastAPI(title="Legal RAG API", version="1.0.0")

# Initialize retriever once
try:
    retriever = LegalRetriever()
except FileNotFoundError as e:
    logger.error("Vector store missing: %s", e)
    retriever = None


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    metadata_filter: Optional[dict] = None


class QueryResponse(BaseModel):
    content: str
    metadata: dict


@app.post("/query", response_model=List[QueryResponse])
def query_legal_docs(request: QueryRequest):
    """
    Query the legal vector database and return relevant chunks.
    """

    if retriever is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    results = retriever.retrieve(
        query=request.query,
        k=request.top_k,
        metadata_filter=request.metadata_filter,
    )

    response = [
        QueryResponse(content=doc.page_content, metadata=doc.metadata)
        for doc in results
    ]

    logger.info("Returned %d results for query '%s'", len(results), request.query)
    return response
