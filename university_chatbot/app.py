"""University Chatbot API with LLM-as-a-Judge quality control."""

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from config.settings import get_settings
from chatbot.query_processor import QueryProcessor
from chatbot.generator import Generator
from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStoreManager
from rag.retriever import Retriever
from judge.evaluator import JudgeEvaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    """Incoming question from a student or staff member."""
    question: str


class AnswerResponse(BaseModel):
    """Structured response including quality evaluation metadata."""
    answer: str
    score: Optional[float] = None
    evaluation: Optional[dict[str, Any]] = None
    passed_quality_check: bool
    attempts: int


# ---------------------------------------------------------------------------
# Application state – populated at startup
# ---------------------------------------------------------------------------

class AppState:
    query_processor: QueryProcessor
    retriever: Retriever
    generator: Generator
    judge: JudgeEvaluator


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise all components on startup, clean up on shutdown."""
    settings = get_settings()

    # Logging
    logging.basicConfig(
        level=getattr(logging, settings.log.level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Query processor (no LLM rewrite for now – keeps startup fast)
    state.query_processor = QueryProcessor()

    # Embeddings → Vector store → Retriever
    embedding_mgr = EmbeddingManager(
        provider=settings.llm.provider,
        model_name=settings.llm.embedding_model,
    )
    vector_store_mgr = VectorStoreManager(
        persist_directory=settings.vector_store.path,
        embedding_function=embedding_mgr.get_embedding_function(),
    )
    state.retriever = Retriever(vector_store_manager=vector_store_mgr)

    # Generator LLM
    state.generator = Generator(
        provider=settings.llm.provider,
        model_name=settings.llm.generator_model,
    )

    # Judge LLM (potentially a stronger model)
    state.judge = JudgeEvaluator(
        provider=settings.llm.provider,
        model_name=settings.llm.judge_model,
    )

    logger.info("All components initialised – server is ready.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="University Chatbot API",
    description=(
        "RAG‑powered university chatbot with LLM‑as‑a‑Judge "
        "quality control and hallucination detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

SAFE_FALLBACK = (
    "I could not verify this information from the available "
    "university sources. Please contact the university directly."
)


@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    """Answer a student/staff question with RAG + Judge quality loop."""
    settings = get_settings()
    max_attempts = settings.judge.max_regeneration_attempts
    threshold = settings.judge.pass_threshold

    # 1. Process the query
    processed = state.query_processor.process(request.question)
    search_query = processed.rewritten_query or processed.cleaned_query

    # 2. Retrieve context
    retrieval = state.retriever.retrieve(search_query)
    if retrieval.num_results == 0:
        return AnswerResponse(
            answer=SAFE_FALLBACK,
            passed_quality_check=False,
            attempts=0,
        )

    # 3. Generate → Judge loop
    last_evaluation: Optional[dict[str, Any]] = None
    last_score: Optional[float] = None

    for attempt in range(1, max_attempts + 1):
        # a. Generate answer
        gen_result = state.generator.generate(
            question=processed.cleaned_query,
            context=retrieval.context,
        )

        # b. Evaluate answer
        eval_result = state.judge.evaluate(
            question=processed.cleaned_query,
            context=retrieval.context,
            answer=gen_result.answer,
        )

        last_score = eval_result.scores.overall_score
        last_evaluation = eval_result.scores.model_dump()

        # c. Quality gate
        if eval_result.passed:
            return AnswerResponse(
                answer=gen_result.answer,
                score=last_score,
                evaluation=last_evaluation,
                passed_quality_check=True,
                attempts=attempt,
            )

        logger.warning(
            "Attempt %d/%d failed quality check (score=%.2f, hallucination=%s)",
            attempt,
            max_attempts,
            last_score or 0,
            eval_result.scores.hallucination,
        )

    # 4. All attempts exhausted → safe fallback
    return AnswerResponse(
        answer=SAFE_FALLBACK,
        score=last_score,
        evaluation=last_evaluation,
        passed_quality_check=False,
        attempts=max_attempts,
    )


@app.get("/health")
async def health():
    """Simple health‑check endpoint."""
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
    )
