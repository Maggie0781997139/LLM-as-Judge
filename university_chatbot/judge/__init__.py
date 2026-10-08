"""LLM-as-a-Judge module for evaluating generated answers."""

from .judge_prompt import JUDGE_SYSTEM_PROMPT, JUDGE_EVALUATION_PROMPT
from .scoring import EvaluationScores, calculate_overall_score, passes_quality_check, DEFAULT_WEIGHTS
from .hallucination import HallucinationReport, analyze_hallucination, format_hallucination_report
from .evaluator import JudgeEvaluator, EvaluationResult

__all__ = [
    "JUDGE_SYSTEM_PROMPT",
    "JUDGE_EVALUATION_PROMPT",
    "EvaluationScores",
    "calculate_overall_score",
    "passes_quality_check",
    "DEFAULT_WEIGHTS",
    "HallucinationReport",
    "analyze_hallucination",
    "format_hallucination_report",
    "JudgeEvaluator",
    "EvaluationResult"
]
