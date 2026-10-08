import logging
from typing import List, Tuple, Any

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages document storage and retrieval using ChromaDB."""

    def __init__(self, persist_directory: str, embedding_function: Any):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function
        )

    def create_from_documents(self, documents: List[Document]) -> None:
        """Creates a new vector store collection from documents."""
        try:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                persist_directory=self.persist_directory,
            )
            logger.info(f"Created vector store with {len(documents)} documents.")
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")

    def add_documents(self, documents: List[Document]) -> None:
        """Adds documents to the existing vector store."""
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store.")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Performs a similarity search."""
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def similarity_search_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Performs a similarity search returning documents with relevance scores."""
        try:
            return self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"Error during similarity search with scores: {e}")
            return []

    def delete_collection(self) -> None:
        """Deletes the entire collection."""
        try:
            self.vector_store.delete_collection()
            logger.info("Deleted vector store collection.")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
