from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.logger import get_logger

logger = get_logger("retriever")
VECTOR_STORE_DIR = Path("vector_store")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class LegalRetriever:
    """Encapsulates FAISS retrieval with HuggingFace embeddings using LangChain v0.3 standards."""

    def __init__(self, vector_store_path: Path = VECTOR_STORE_DIR, top_k: int = 5):
        if not vector_store_path.exists():
            raise FileNotFoundError(f"Vector store not found at {vector_store_path}")

        # Load embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.top_k = top_k

        # Load vector database safely
        try:
            logger.info("Loading FAISS vector store from %s", vector_store_path)
            # allow_dangerous_deserialization is required for loading local pickle files
            self.vector_db = FAISS.load_local(
                str(vector_store_path),
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            logger.exception("Failed to load vector store")
            raise e

        logger.info("LegalRetriever initialized successfully")

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        metadata_filter: Optional[dict] = None,
    ) -> List[Document]:
        """
        Retrieve relevant legal documents.

        Uses the direct similarity_search method to ensure thread-safety
        when handling concurrent API requests.
        """
        search_k = k or self.top_k
        logger.info("Running retrieval for query: '%s' (k=%d)", query, search_k)

        try:
            # Using similarity_search directly is more robust for dynamic filtering
            results = self.vector_db.max_marginal_relevance_search(
                query=query,
                k=search_k,
                fetch_k=20,  # Fetch 20, then pick the 5 most diverse
                filter=metadata_filter,
            )

            logger.info("Retrieved %d documents", len(results))
            return results

        except Exception as e:
            logger.error("Error during retrieval: %s", str(e))
            return []

    async def aretrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Asynchronous version for high-concurrency API environments."""
        # FAISS search is CPU-bound, so this usually runs in a thread pool via LangChain
        search_k = k or self.top_k
        return await self.vector_db.asimilarity_search(query=query, k=search_k)
