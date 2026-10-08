import logging
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class TextChunker:
    """Chunks text and documents into smaller pieces."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of Document objects into chunks."""
        try:
            chunks = self.splitter.split_documents(documents)
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
            return chunks
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            return []

    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Splits a raw text string into Document chunks."""
        try:
            meta = metadata or {}
            chunks = self.splitter.create_documents([text], metadatas=[meta])
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            return []
