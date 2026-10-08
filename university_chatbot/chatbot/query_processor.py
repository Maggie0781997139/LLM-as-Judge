"""Query processing for the university chatbot."""

import unicodedata
import logging
from typing import Optional
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from .prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)

class ProcessedQuery(BaseModel):
    """Model representing a processed query."""
    original_query: str
    cleaned_query: str
    rewritten_query: Optional[str] = None
    intent: Optional[str] = None

class QueryProcessor:
    """Processes, cleans, and rewrites student queries."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Initialize the QueryProcessor.

        Args:
            llm: Optional LangChain ChatModel instance for query rewriting.
        """
        self.llm = llm

    def clean_query(self, query: str) -> str:
        """
        Perform basic text cleaning on the query.

        Args:
            query: The original query string.

        Returns:
            The cleaned query string.
        """
        # Strip whitespace
        cleaned = query.strip()
        # Normalize unicode
        cleaned = unicodedata.normalize("NFKC", cleaned)
        return cleaned

    def detect_intent(self, query: str) -> str:
        """
        Detect the intent of the query based on simple keyword matching.

        Args:
            query: The cleaned query string.

        Returns:
            The detected intent category.
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["apply", "admission", "requirements", "deadline", "enroll"]):
            return "admission"
        elif any(word in query_lower for word in ["course", "class", "major", "degree", "credits", "curriculum", "academic"]):
            return "academic"
        elif any(word in query_lower for word in ["tuition", "fee", "cost", "pay", "scholarship", "financial aid", "finaid"]):
            return "fees"
        elif any(word in query_lower for word in ["rules", "policy", "regulation", "honor code", "conduct"]):
            return "regulations"
        elif any(word in query_lower for word in ["housing", "dorm", "dining", "health", "library", "career", "service"]):
            return "student_services"
        elif any(word in query_lower for word in ["department", "faculty", "professor", "contact", "email"]):
            return "departments"
        
        return "general"

    def process(self, query: str) -> ProcessedQuery:
        """
        Process the query by cleaning, detecting intent, and optionally rewriting.

        Args:
            query: The original query string.

        Returns:
            A ProcessedQuery object containing the processed results.
        """
        cleaned = self.clean_query(query)
        intent = self.detect_intent(cleaned)
        
        rewritten = None
        if self.llm:
            try:
                prompt = QUERY_REWRITE_PROMPT.format(question=cleaned)
                response = self.llm.invoke([HumanMessage(content=prompt)])
                rewritten = response.content.strip()
                logger.debug(f"Rewrote query: '{cleaned}' -> '{rewritten}'")
            except Exception as e:
                logger.error(f"Failed to rewrite query: {e}")

        return ProcessedQuery(
            original_query=query,
            cleaned_query=cleaned,
            rewritten_query=rewritten,
            intent=intent
        )
