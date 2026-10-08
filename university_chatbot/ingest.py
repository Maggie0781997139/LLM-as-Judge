"""
Document Ingestion Script

Loads university documents from the knowledge_base/ directory,
chunks them, generates embeddings, and stores them in ChromaDB.

Usage:
    python ingest.py                          # Ingest all knowledge_base/
    python ingest.py --source knowledge_base/admissions
    python ingest.py --source docs/my_file.pdf
    python ingest.py --reset                  # Wipe vector store and re-ingest
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config.settings import get_settings
from rag.document_loader import DocumentLoader
from rag.chunker import TextChunker
from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStoreManager

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the ingestion process."""
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def count_files(directory: Path) -> dict[str, int]:
    """Count files by extension in a directory tree."""
    counts: dict[str, int] = {}
    for f in directory.rglob("*"):
        if f.is_file() and f.suffix.lower() in {".pdf", ".docx", ".txt"}:
            ext = f.suffix.lower()
            counts[ext] = counts.get(ext, 0) + 1
    return counts


def print_scan_results(source: Path, counts: dict[str, int]) -> None:
    """Display a summary table of discovered files."""
    table = Table(title=f"📂 Documents found in [bold]{source}[/bold]")
    table.add_column("Type", style="cyan")
    table.add_column("Count", justify="right", style="green")

    total = 0
    for ext in sorted(counts):
        table.add_row(ext, str(counts[ext]))
        total += counts[ext]

    table.add_section()
    table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")
    console.print(table)


def ingest(
    source: Path,
    reset: bool = False,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> None:
    """Run the full ingestion pipeline."""

    settings = get_settings()
    _chunk_size = chunk_size or settings.vector_store.chunk_size
    _chunk_overlap = chunk_overlap or settings.vector_store.chunk_overlap

    # ------------------------------------------------------------------
    # 1. Scan source
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 1 · Scanning documents")

    if source.is_file():
        counts = {source.suffix.lower(): 1}
    else:
        counts = count_files(source)

    if not counts:
        console.print(
            f"[bold red]No supported files (.pdf, .docx, .txt) found in {source}[/bold red]"
        )
        console.print(
            "\n[dim]Place your university documents in the knowledge_base/ "
            "subdirectories and run this script again.[/dim]"
        )
        sys.exit(1)

    print_scan_results(source, counts)

    # ------------------------------------------------------------------
    # 2. Load documents
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 2 · Loading documents")

    loader = DocumentLoader()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading files…", total=None)
        if source.is_file():
            documents = loader.load_file(str(source))
        else:
            documents = loader.load_directory(str(source))

    console.print(f"  Loaded [green]{len(documents)}[/green] document pages/sections")

    if not documents:
        console.print("[bold red]No content extracted. Check your files.[/bold red]")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Chunk documents
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 3 · Chunking documents")

    chunker = TextChunker(chunk_size=_chunk_size, chunk_overlap=_chunk_overlap)
    chunks = chunker.chunk_documents(documents)

    console.print(
        f"  Created [green]{len(chunks)}[/green] chunks "
        f"(size={_chunk_size}, overlap={_chunk_overlap})"
    )

    # ------------------------------------------------------------------
    # 4. Generate embeddings & store
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 4 · Embedding & storing in ChromaDB")

    embedding_mgr = EmbeddingManager(
        provider=settings.llm.provider,
        model_name=settings.llm.embedding_model,
    )
    embedding_fn = embedding_mgr.get_embedding_function()

    vector_store = VectorStoreManager(
        persist_directory=settings.vector_store.path,
        embedding_function=embedding_fn,
    )

    if reset:
        console.print("  [yellow]Resetting vector store…[/yellow]")
        vector_store.delete_collection()
        # Re-create after deletion
        vector_store = VectorStoreManager(
            persist_directory=settings.vector_store.path,
            embedding_function=embedding_fn,
        )

    start = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        # Process in batches to show progress and avoid API rate limits
        batch_size = 50
        task = progress.add_task("Embedding chunks…", total=len(chunks))

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            if i == 0 and reset:
                vector_store.create_from_documents(batch)
            else:
                vector_store.add_documents(batch)
            progress.update(task, advance=len(batch))

    elapsed = time.time() - start

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    console.rule("[bold green]✅ Ingestion complete")

    summary = Table(title="Ingestion Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right", style="green")
    summary.add_row("Source", str(source))
    summary.add_row("Documents loaded", str(len(documents)))
    summary.add_row("Chunks created", str(len(chunks)))
    summary.add_row("Chunk size", str(_chunk_size))
    summary.add_row("Chunk overlap", str(_chunk_overlap))
    summary.add_row("Vector store", settings.vector_store.path)
    summary.add_row("Embedding provider", settings.llm.provider)
    summary.add_row("Embedding model", settings.llm.embedding_model)
    summary.add_row("Time elapsed", f"{elapsed:.1f}s")
    console.print(summary)

    console.print(
        "\n[dim]You can now start the API server with:[/dim] "
        "[bold]python app.py[/bold]\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest university documents into the vector store.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="knowledge_base",
        help="Directory or file to ingest (default: knowledge_base/)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing vector store before ingesting",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size from settings",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override chunk overlap from settings",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    source = Path(args.source)
    if not source.exists():
        console.print(f"[bold red]Source not found: {source}[/bold red]")
        sys.exit(1)

    ingest(
        source=source,
        reset=args.reset,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
