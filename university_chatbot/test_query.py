"""
Test script to send queries to the University Chatbot API.
Make sure `python app.py` is running before using this script.

Usage:
    python test_query.py "What are the admission requirements?"
    python test_query.py --interactive
"""

import argparse
import sys
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

console = Console()
API_URL = "http://localhost:8000/ask"

def send_query(question: str):
    """Sends a question to the FastAPI server and pretty-prints the result."""
    console.print(f"\n[bold blue]Sending query:[/bold blue] {question}")
    
    with console.status("Waiting for RAG + Judge Evaluation...", spinner="dots"):
        try:
            # We set a high timeout because generating and judging can take 5-15 seconds
            response = httpx.post(
                API_URL, 
                json={"question": question},
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
        except httpx.ConnectError:
            console.print("[bold red]Error: Could not connect to the API.[/bold red]")
            console.print("Make sure you have started the server with: [bold]python app.py[/bold]")
            return
        except httpx.TimeoutException:
            console.print("[bold red]Error: Request timed out. The LLMs took too long to respond.[/bold red]")
            return
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            return

    # Print the Answer
    console.print(Panel(Markdown(data["answer"]), title="Generated Answer", border_style="green" if data["passed_quality_check"] else "red"))
    
    # Print the Judge's Evaluation details if available
    evaluation = data.get("evaluation")
    if evaluation:
        console.print("\n[bold]⚖️ LLM-as-a-Judge Evaluation[/bold]")
        
        # Summary metrics
        score_color = "green" if data["passed_quality_check"] else "red"
        console.print(f"Overall Score: [bold {score_color}]{data['score']}/10[/bold {score_color}]")
        console.print(f"Attempts: {data['attempts']}")
        console.print(f"Hallucination Detected: [{'bold red' if evaluation['hallucination'] else 'bold green'}]{evaluation['hallucination']}[/]")
        
        if evaluation.get('hallucinated_claims'):
            console.print("[bold red]Hallucinated Claims:[/bold red]")
            for claim in evaluation['hallucinated_claims']:
                console.print(f"  - {claim}")
        
        console.print(f"[bold]Judge's Reasoning:[/bold] {evaluation['reason']}\n")
        
        # Detailed score breakdown table
        table = Table(show_header=True, header_style="cyan")
        table.add_column("Criterion")
        table.add_column("Score (1-10)")
        table.add_column("Weight")
        
        table.add_row("Faithfulness", str(evaluation["faithfulness"]), "30%")
        table.add_row("Correctness", str(evaluation["correctness"]), "25%")
        table.add_row("Relevance", str(evaluation["relevance"]), "20%")
        table.add_row("Completeness", str(evaluation["completeness"]), "15%")
        table.add_row("Clarity", str(evaluation["clarity"]), "10%")
        
        console.print(table)
    else:
        console.print("[yellow]No evaluation metadata was returned.[/yellow]")

def main():
    parser = argparse.ArgumentParser(description="Test the University Chatbot API")
    parser.add_argument("question", nargs="?", type=str, help="The question to ask")
    parser.add_argument("-i", "--interactive", action="store_true", help="Start interactive chat mode")
    
    args = parser.parse_args()
    
    if args.interactive or not args.question:
        console.print("[bold green]University Chatbot - Interactive Testing[/bold green]")
        console.print("Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                question = console.input("[bold cyan]You:[/bold cyan] ")
                if question.lower() in ["exit", "quit"]:
                    break
                if question.strip():
                    send_query(question)
            except KeyboardInterrupt:
                break
    else:
        send_query(args.question)

if __name__ == "__main__":
    main()
