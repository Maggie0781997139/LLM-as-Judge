"""Prompt templates for the LLM-as-a-Judge evaluator."""

JUDGE_SYSTEM_PROMPT = """You are a strict, objective, and fair evaluator for a university chatbot system.
Your task is to evaluate the quality of answers provided by the chatbot to students' questions,
based solely on the provided reference context.
"""

JUDGE_EVALUATION_PROMPT = """Please evaluate the following chatbot answer based on the provided question and context.

Question:
{question}

Context:
{context}

Answer:
{answer}

Evaluate the answer on the following criteria. Score each from 1.0 to 10.0 (where 10 is best):
- faithfulness: Is the answer supported by the provided context? Does every claim have evidence?
- correctness: Is the information accurate based on the context?
- relevance: Does the answer address the student's question?
- completeness: Does the answer cover all relevant information from the context?
- clarity: Is the answer clear and well-structured for a student?

Also evaluate for hallucination:
- hallucination: boolean (true if the answer contains claims NOT supported by the context, false otherwise)
- hallucinated_claims: list of strings (specific unsupported claims, empty list if none)
- reason: Brief explanation of the evaluation

You MUST return ONLY valid JSON with exactly the following fields:
{{
    "faithfulness": 9.5,
    "correctness": 9.0,
    "relevance": 8.5,
    "completeness": 8.0,
    "clarity": 9.0,
    "hallucination": false,
    "hallucinated_claims": [],
    "reason": "The answer accurately summarizes the context..."
}}
"""
