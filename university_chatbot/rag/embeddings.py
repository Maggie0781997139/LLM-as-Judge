import logging
from typing import Optional, Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embedding functions from different providers."""

    def __init__(self, provider: str = 'google', model_name: Optional[str] = None):
        self.provider = provider.lower()
        self.model_name = model_name
        self._embedding_function = None

    def get_embedding_function(self) -> Any:
        """Returns the appropriate LangChain embedding instance via lazy initialization."""
        if self._embedding_function is not None:
            return self._embedding_function

        try:
            if self.provider == 'google':
                model = self.model_name or "models/embedding-001"
                self._embedding_function = GoogleGenerativeAIEmbeddings(model=model)
            elif self.provider == 'openai':
                model = self.model_name or "text-embedding-3-small"
                self._embedding_function = OpenAIEmbeddings(model=model)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            return self._embedding_function
        except Exception as e:
            logger.error(f"Error initializing embedding function for provider {self.provider}: {e}")
            raise
