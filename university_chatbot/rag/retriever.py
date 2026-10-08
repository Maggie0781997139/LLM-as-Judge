import logging
from typing import List, Tuple

from pydantic import BaseModel, Field
from langchain_core.documents import Document

from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class RetrievalResult(BaseModel):
    """Pydantic model representing a retrieval result."""
    query: str = Field(..., description="The query used for retrieval")
    documents: List[Document] = Field(..., description="The retrieved documents")
    context: str = Field(..., description="The formatted context from documents")
    num_results: int = Field(..., description="Number of documents retrieved")

    class Config:
        arbitrary_types_allowed = True

class Retriever:
    """Retrieves relevant context for queries."""

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """Retrieves relevant documents and formats them into a result object."""
        try:
            docs = self.vector_store_manager.similarity_search(query, k=k)
            context = "\n\n".join(doc.page_content for doc in docs)
            
            result = RetrievalResult(
                query=query,
                documents=docs,
                context=context,
                num_results=len(docs)
            )
            logger.info(f"Retrieved {len(docs)} documents for query: '{query}'")
            return result
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return RetrievalResult(query=query, documents=[], context="", num_results=0)

    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieves relevant documents along with their similarity scores."""
        try:
            results = self.vector_store_manager.similarity_search_with_scores(query, k=k)
            logger.info(f"Retrieved {len(results)} scored documents for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"Error during scored retrieval: {e}")
            return []
