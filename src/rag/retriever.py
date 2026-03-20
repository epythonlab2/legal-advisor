from pathlib import Path
from typing import List

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.logger import get_logger

logger = get_logger("retriever")
VECTOR_STORE_DIR = Path("vector_store")
# Primary embedding model (Bi-Encoder)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class LegalRetriever:
    def __init__(self, vector_store_path: Path = VECTOR_STORE_DIR, top_k: int = 5):
        if not vector_store_path.exists():
            raise FileNotFoundError(f"Vector store not found at {vector_store_path}")

        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.top_k = top_k

        try:
            logger.info("Loading FAISS vector store...")
            self.vector_db = FAISS.load_local(
                str(vector_store_path),
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True,
            )

            # 1. Initialize the Reranker (Cross-Encoder)
            # This model is specifically trained to rank document relevance
            compressor = FlashrankRerank(model="ms-marco-MultiBERT-L-12", top_n=top_k)

            # 2. Setup the Base Retriever
            # We fetch more candidates (e.g., 20) to give the reranker a good selection
            base_retriever = self.vector_db.as_retriever(search_kwargs={"k": 20})

            # 3. Create the Compression Retriever
            # This wraps the base retriever and applies the reranking logic automatically
            self.compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=base_retriever
            )

        except Exception as e:
            logger.exception("Failed to load vector store")
            raise e

        logger.info("LegalRetriever with Reranking initialized")

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve and Rerank documents.
        The reranker will filter out 'noise' like the waste management snippets
        if they don't semantically answer the query 'What is a contract?'.
        """
        logger.info("Running Reranked Retrieval for: '%s'", query)
        try:
            # invoke() handles the full pipeline: Retrieve 20 -> Rerank -> Return top_k
            results = self.compression_retriever.invoke(query)

            logger.info("Retrieved %d reranked documents", len(results))
            return results

        except Exception as e:
            logger.error("Error during retrieval: %s", str(e))
            return []
