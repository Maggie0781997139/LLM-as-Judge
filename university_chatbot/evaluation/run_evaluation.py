import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table

console = Console()

from config.settings import get_settings
from chatbot.query_processor import QueryProcessor
from chatbot.generator import Generator
from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStoreManager
from rag.retriever import Retriever
from judge.evaluator import JudgeEvaluator

# Initialize global pipeline components lazily
_pipeline_components = {}

def get_components():
    """Lazy-loads the real pipeline components for evaluation."""
    if not _pipeline_components:
        settings = get_settings()
        embedding_mgr = EmbeddingManager(
            provider=settings.llm.provider,
            model_name=settings.llm.embedding_model,
        )
        vector_store_mgr = VectorStoreManager(
            persist_directory=settings.vector_store.path,
            embedding_function=embedding_mgr.get_embedding_function(),
        )
        _pipeline_components["processor"] = QueryProcessor()
        _pipeline_components["retriever"] = Retriever(vector_store_manager=vector_store_mgr)
        _pipeline_components["generator"] = Generator(
            provider=settings.llm.provider,
            model_name=settings.llm.generator_model,
        )
        _pipeline_components["judge"] = JudgeEvaluator(
            provider=settings.llm.provider,
            model_name=settings.llm.judge_model,
        )
    return _pipeline_components

def run_pipeline(question: str, expected_answer: str, threshold: float) -> Dict[str, Any]:
    """Runs the real retrieve -> generate -> judge pipeline on a test case."""
    comps = get_components()
    
    # 1. Process
    processed = comps["processor"].process(question)
    search_query = processed.rewritten_query or processed.cleaned_query
    
    # 2. Retrieve
    retrieval = comps["retriever"].retrieve(search_query)
    
    # 3. Generate
    gen_result = comps["generator"].generate(
        question=processed.cleaned_query,
        context=retrieval.context,
    )
    
    # 4. Judge
    eval_result = comps["judge"].evaluate(
        question=processed.cleaned_query,
        context=retrieval.context,
        answer=gen_result.answer,
    )
    
    # Add passed status based on CLI threshold (overriding app threshold)
    passed = eval_result.scores.overall_score >= threshold and not eval_result.scores.hallucination
    
    eval_dict = eval_result.scores.model_dump()
    eval_dict["passed"] = passed
    
    return {
        "question": question,
        "expected_answer": expected_answer,
        "generated_answer": gen_result.answer,
        "context_used": retrieval.context,
        "evaluation": eval_dict
    }

def print_summary(results: List[Dict[str, Any]], stats: Dict[str, Any]):
    """Prints a summary table using rich."""
    console.print("\n[bold]Evaluation Summary[/bold]")
    console.print(f"Total Cases: {stats['total_cases']}")
    console.print(f"Pass Rate: {stats['pass_rate']:.1f}%")
    console.print(f"Hallucination Rate: {stats['hallucination_rate']:.1f}%")
    console.print(f"Average Score: {stats['mean_score']:.2f}\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=4)
    table.add_column("Question")
    table.add_column("Score", justify="right")
    table.add_column("Passed", justify="center")
    table.add_column("Hallucination", justify="center")

    for i, res in enumerate(results):
        eval_data = res["evaluation"]
        passed_str = "[green]Yes[/green]" if eval_data["passed"] else "[red]No[/red]"
        halluc_str = "[red]Yes[/red]" if eval_data["hallucination"] else "[green]No[/green]"
        table.add_row(
            str(i+1), 
            res["question"], 
            f"{eval_data['overall_score']:.2f}", 
            passed_str, 
            halluc_str
        )
    
    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Run batch evaluation for the university chatbot.")
    parser.add_argument("--dataset", type=str, default="evaluation/dataset.json", help="Path to evaluation dataset")
    parser.add_argument("--output-dir", type=str, default="evaluation/results", help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=0.7, help="Passing score threshold")
    args = parser.parse_args()

    console.print(f"[bold blue]Starting evaluation...[/bold blue]")
    console.print(f"Dataset: {args.dataset}")
    console.print(f"Threshold: {args.threshold}")

    if not os.path.exists(args.dataset):
        console.print(f"[bold red]Error: Dataset file '{args.dataset}' not found.[/bold red]")
        return

    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    results = []
    passed_count = 0
    hallucination_count = 0
    total_score = 0.0

    for item in dataset:
        result = run_pipeline(item["question"], item["expected_answer"], args.threshold)
        results.append(result)
        
        if result["evaluation"]["passed"]:
            passed_count += 1
        if result["evaluation"]["hallucination"]:
            hallucination_count += 1
        total_score += result["evaluation"]["overall_score"]

    total_cases = len(dataset)
    stats = {
        "total_cases": total_cases,
        "pass_rate": (passed_count / total_cases) * 100 if total_cases > 0 else 0,
        "hallucination_rate": (hallucination_count / total_cases) * 100 if total_cases > 0 else 0,
        "mean_score": total_score / total_cases if total_cases > 0 else 0
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"eval_results_{timestamp}.json")
    
    with open(output_file, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "dataset": args.dataset,
                "threshold": args.threshold,
                "stats": stats
            },
            "results": results
        }, f, indent=4)

    console.print(f"\n[green]Results saved to {output_file}[/green]")
    print_summary(results, stats)

if __name__ == "__main__":
    main()
