"""Main LLM-as-a-Judge evaluator implementation."""
import json
import logging
import re
from typing import Optional
from pydantic import BaseModel

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from .judge_prompt import JUDGE_SYSTEM_PROMPT, JUDGE_EVALUATION_PROMPT
from .scoring import EvaluationScores, calculate_overall_score, passes_quality_check
from .hallucination import HallucinationReport, analyze_hallucination

logger = logging.getLogger(__name__)

class EvaluationResult(BaseModel):
    """Complete evaluation result from the judge."""
    scores: EvaluationScores
    hallucination_report: HallucinationReport
    passed: bool
    raw_response: str

class JudgeEvaluator:
    """Evaluates chatbot answers using an LLM as a judge."""

    def __init__(self, provider: str = 'openai', model_name: str = 'gpt-4o'):
        """Initialize the evaluator with a specific LLM provider and model."""
        self.provider = provider.lower()
        self.model_name = model_name
        
        if self.provider == 'openai':
            self.llm = ChatOpenAI(model=self.model_name, temperature=0.0)
        elif self.provider == 'google':
            self.llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=0.0)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'google'.")
            
    def _parse_judge_response(self, response: str) -> EvaluationScores:
        """Robustly parse JSON from the LLM response."""
        try:
            # Attempt direct JSON parsing first
            data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback to extracting JSON from markdown or text
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from markdown: {e}")
                    raise ValueError("Invalid JSON in judge response") from e
            else:
                # Find the first { and last }
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    try:
                        data = json.loads(response[start_idx:end_idx+1])
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to extract and parse JSON: {e}")
                        raise ValueError("Could not extract valid JSON from response") from e
                else:
                    raise ValueError("No JSON object found in response")
        
        # Create and validate the Pydantic model
        return EvaluationScores(**data)

    def evaluate(self, question: str, context: str, answer: str) -> EvaluationResult:
        """Evaluates an answer given the question and reference context."""
        logger.info("Starting evaluation for question: %s", question[:50])
        
        # 1. Build the judge prompt
        human_prompt = JUDGE_EVALUATION_PROMPT.format(
            question=question,
            context=context,
            answer=answer
        )
        
        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=human_prompt)
        ]
        
        # 2. Call the judge LLM
        try:
            logger.debug("Calling LLM judge (%s, %s)", self.provider, self.model_name)
            llm_response = self.llm.invoke(messages)
            response_text = str(llm_response.content)
        except Exception as e:
            logger.error(f"Error calling LLM judge: {e}")
            raise RuntimeError(f"Judge LLM invocation failed: {e}") from e

        # 3. Parse the JSON response
        try:
            scores = self._parse_judge_response(response_text)
        except Exception as e:
            logger.error(f"Failed to parse response: {response_text}")
            raise RuntimeError("Failed to parse evaluation scores") from e

        # 4. Calculate overall score
        scores.overall_score = calculate_overall_score(scores)
        
        # 5. Generate hallucination report
        hallucination_report = analyze_hallucination(scores)
        
        # 6. Determine if it passed quality check
        passed = passes_quality_check(scores)
        
        # 7. Return EvaluationResult
        result = EvaluationResult(
            scores=scores,
            hallucination_report=hallucination_report,
            passed=passed,
            raw_response=response_text
        )
        
        logger.info("Evaluation complete. Passed: %s, Overall Score: %s", passed, scores.overall_score)
        return result
