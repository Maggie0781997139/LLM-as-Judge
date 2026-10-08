"""Scoring models and functions for evaluation."""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator

class EvaluationScores(BaseModel):
    """Pydantic model representing the judge's evaluation scores."""
    faithfulness: float = Field(..., ge=1.0, le=10.0)
    correctness: float = Field(..., ge=1.0, le=10.0)
    relevance: float = Field(..., ge=1.0, le=10.0)
    completeness: float = Field(..., ge=1.0, le=10.0)
    clarity: float = Field(..., ge=1.0, le=10.0)
    hallucination: bool
    hallucinated_claims: List[str]
    reason: str
    overall_score: Optional[float] = None

    @field_validator('faithfulness', 'correctness', 'relevance', 'completeness', 'clarity')
    @classmethod
    def check_score_range(cls, v: float) -> float:
        if not 1.0 <= v <= 10.0:
            raise ValueError(f"Score must be between 1.0 and 10.0, got {v}")
        return v

DEFAULT_WEIGHTS: Dict[str, float] = {
    "faithfulness": 0.30,
    "correctness": 0.25,
    "relevance": 0.20,
    "completeness": 0.15,
    "clarity": 0.10
}

def calculate_overall_score(scores: EvaluationScores, weights: Optional[Dict[str, float]] = None) -> float:
    """Calculates weighted average of scores, rounds to 2 decimal places."""
    w = weights or DEFAULT_WEIGHTS
    
    total_weight = sum(w.values())
    
    weighted_sum = (
        scores.faithfulness * w.get("faithfulness", 0.30) +
        scores.correctness * w.get("correctness", 0.25) +
        scores.relevance * w.get("relevance", 0.20) +
        scores.completeness * w.get("completeness", 0.15) +
        scores.clarity * w.get("clarity", 0.10)
    )
    
    if total_weight > 0:
        weighted_sum /= total_weight
        
    return round(weighted_sum, 2)

def passes_quality_check(scores: EvaluationScores, threshold: float = 8.0) -> bool:
    """Returns True if overall_score >= threshold AND hallucination is False."""
    if scores.overall_score is None:
        raise ValueError("overall_score must be calculated before checking quality")
    return scores.overall_score >= threshold and not scores.hallucination
