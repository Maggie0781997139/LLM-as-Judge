import logging
from datetime import datetime
from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Loads documents from various file formats."""

    def __init__(self):
        pass

    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Loads a single file into Document objects."""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            return []

        try:
            if path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(path))
            elif path.suffix.lower() in ('.docx', '.doc'):
                loader = Docx2txtLoader(str(path))
            elif path.suffix.lower() == '.txt':
                loader = TextLoader(str(path))
            else:
                logger.warning(f"Unsupported file type: {path.suffix}")
                return []

            documents = loader.load()
            timestamp = datetime.now().isoformat()
            
            for doc in documents:
                doc.metadata.update({
                    "source": str(path),
                    "file_type": path.suffix.lower(),
                    "load_timestamp": timestamp
                })
            
            return documents
        except Exception as e:
            logger.error(f"Error loading file {path}: {e}")
            return []

    def load_directory(self, directory_path: Union[str, Path]) -> List[Document]:
        """Recursively loads all supported files from a directory."""
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            logger.error(f"Directory not found or invalid: {path}")
            return []

        all_documents = []
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ('.pdf', '.docx', '.doc', '.txt'):
                docs = self.load_file(file_path)
                all_documents.extend(docs)

        return all_documents
