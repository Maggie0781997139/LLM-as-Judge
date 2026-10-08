"""Hallucination detection utilities."""
from typing import List
from pydantic import BaseModel
from .scoring import EvaluationScores

class HallucinationReport(BaseModel):
    """Report detailing any hallucinations found in the answer."""
    has_hallucination: bool
    hallucinated_claims: List[str]
    severity: str  # 'none', 'low', 'medium', 'high'
    recommendation: str

def analyze_hallucination(evaluation: EvaluationScores) -> HallucinationReport:
    """Determines hallucination severity and provides recommendations."""
    if not evaluation.hallucination:
        return HallucinationReport(
            has_hallucination=False,
            hallucinated_claims=[],
            severity="none",
            recommendation="No hallucinations detected. Proceed with the answer."
        )

    severity = "none"
    recommendation = ""
    
    f_score = evaluation.faithfulness
    if f_score >= 7.0:
        severity = "low"
        recommendation = "Minor unsupported claims. Review and slightly edit the answer."
    elif 4.0 <= f_score < 7.0:
        severity = "medium"
        recommendation = "Significant unsupported claims. Extensive review or regeneration recommended."
    else:
        severity = "high"
        recommendation = "Severe hallucinations detected. Regenerate the answer immediately."

    return HallucinationReport(
        has_hallucination=True,
        hallucinated_claims=evaluation.hallucinated_claims,
        severity=severity,
        recommendation=recommendation
    )

def format_hallucination_report(report: HallucinationReport) -> str:
    """Returns a human-readable string representation of the report."""
    if not report.has_hallucination:
        return "Hallucination Status: Passed (No hallucinations detected)."
    
    lines = [
        f"Hallucination Status: Failed (Severity: {report.severity.upper()})",
        f"Recommendation: {report.recommendation}",
        "Hallucinated Claims:"
    ]
    for claim in report.hallucinated_claims:
        lines.append(f" - {claim}")
        
    return "\n".join(lines)
