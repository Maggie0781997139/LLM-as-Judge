"""Answer generation for the university chatbot."""

import logging
from typing import Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from .prompts import SYSTEM_PROMPT, GENERATOR_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class GenerationResult(BaseModel):
    """Model representing the result of answer generation."""
    answer: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

class Generator:
    """Generates answers to queries using retrieved context."""

    def __init__(self, provider: str, model_name: str):
        """
        Initialize the Generator.

        Args:
            provider: The LLM provider ('google' or 'openai').
            model_name: The name of the model to use.
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initializes the appropriate LangChain ChatModel."""
        if self.provider == "openai":
            return ChatOpenAI(model=self.model_name, temperature=0.0)
        elif self.provider == "google":
            return ChatGoogleGenerativeAI(model=self.model_name, temperature=0.0)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}. Must be 'openai' or 'google'.")

    def generate(self, question: str, context: str) -> GenerationResult:
        """
        Generate an answer to the question using the provided context.

        Args:
            question: The user's question.
            context: The retrieved context documents.

        Returns:
            A GenerationResult object containing the answer and metadata.
        """
        prompt_content = GENERATOR_PROMPT_TEMPLATE.format(context=context, question=question)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_content)
        ]

        logger.info(f"Generating answer for question: '{question}' using model: {self.model_name}")

        try:
            response = self.llm.invoke(messages)
            
            prompt_tokens = None
            completion_tokens = None
            
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                prompt_tokens = response.usage_metadata.get("input_tokens")
                completion_tokens = response.usage_metadata.get("output_tokens")
            elif hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
                token_usage = response.response_metadata["token_usage"]
                prompt_tokens = token_usage.get("prompt_tokens")
                completion_tokens = token_usage.get("completion_tokens")

            return GenerationResult(
                answer=response.content,
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            fallback_answer = "I'm sorry, I encountered an error while trying to generate an answer. Please try again later."
            return GenerationResult(
                answer=fallback_answer,
                model=self.model_name
            )
