from datetime import datetime
from typing import List, Optional
from sqlalchemy import String, Text, Float, Boolean, Integer, DateTime, ForeignKey, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.engine import Engine
import json

class Base(DeclarativeBase):
    pass

class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    processed_query: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    context_retrieved: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    evaluations: Mapped[List["Evaluation"]] = relationship(back_populates="conversation")

class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"))
    faithfulness: Mapped[float] = mapped_column(Float)
    correctness: Mapped[float] = mapped_column(Float)
    relevance: Mapped[float] = mapped_column(Float)
    completeness: Mapped[float] = mapped_column(Float)
    clarity: Mapped[float] = mapped_column(Float)
    overall_score: Mapped[float] = mapped_column(Float)
    hallucination: Mapped[bool] = mapped_column(Boolean)
    hallucinated_claims: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reason: Mapped[str] = mapped_column(Text)
    passed: Mapped[bool] = mapped_column(Boolean)
    attempt_number: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    conversation: Mapped["Conversation"] = relationship(back_populates="evaluations")

def get_engine(database_url: str) -> Engine:
    """Create database engine."""
    return create_engine(database_url)

def create_tables(engine: Engine):
    """Create all tables in the database."""
    Base.metadata.create_all(engine)

def get_session(engine: Engine) -> sessionmaker:
    """Return a session maker for the given engine."""
    return sessionmaker(bind=engine)
