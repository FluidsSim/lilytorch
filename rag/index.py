"""
rag/index.py — Index the lilytorch codebase AND research papers into ChromaDB.

THREE KINDS OF KNOWLEDGE
────────────────────────
1. **Code** — your .py files: solver, body, advection, Poisson, integration, etc.
2. **Docs** — your .rst/.md files: mathematical formulation, numerical schemes, API.
3. **Papers** — PDF files you drop into rag/papers/: BDIM papers, IBM methods,
   CFD textbooks, multi-rigid-body dynamics references, anything you want.

Each source gets tagged with metadata (source_type = "code" | "docs" | "paper")
so at query time we can filter or weight them differently.

USAGE
─────
    python rag/index.py                     # index code + docs + papers
    python rag/index.py --force             # re-index everything from scratch
    python rag/index.py --papers-only       # only re-index the papers folder
    python rag/index.py --include-farms     # also index FARMS submodules

PDF PAPERS
──────────
Drop your PDFs into  rag/papers/  — subdirectories are fine:
    rag/papers/
        weymouth_yue_2011_bdim.pdf
        maertens_weymouth_2015.pdf
        coquerelle_cottet_2008.pdf
        textbooks/
            anderson_cfd_basics_ch3.pdf
        my_notes/
            bdim_derivation.pdf

The filename becomes part of the metadata, so use descriptive names.
"""

import argparse
import os
import sys
import time
from pathlib import Path

from chromadb import PersistentClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from rich.console import Console
from rich.progress import track

# ── Configuration ─────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent          # lilytorch/
RAG_DIR     = Path(__file__).resolve().parent                  # rag/
PAPERS_DIR  = RAG_DIR / "papers"                               # rag/papers/
DB_DIR      = RAG_DIR / "vectorstore"                          # rag/vectorstore/
COLLECTION  = "lilytorch"

SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".tox", "lilytorch.egg-info",
    "_build", "old", "rag",   # don't index the RAG code itself
}

# Local embedding model — no API key needed, ~80 MB download on first run.
# This model maps text → 384-dimensional vectors.  Texts with similar meaning
# end up close together in this vector space (measured by cosine similarity).
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

console = Console()


# ── 1. Collect source files ──────────────────────────────────────────────────

def collect_code_and_docs(include_farms: bool = False) -> list[dict]:
    """
    Walk the repo tree and collect .py / .rst / .md files.
    Returns a list of dicts with 'path' and 'source_type'.
    """
    extensions = {".py", ".rst", ".md"}
    files = []

    for root, dirs, filenames in os.walk(REPO_ROOT):
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS
            and not (d == "FARMS_V2" and not include_farms)
        ]
        for fname in filenames:
            p = Path(root) / fname
            if p.suffix in extensions:
                stype = "code" if p.suffix == ".py" else "docs"
                files.append({"path": p, "source_type": stype})

    return sorted(files, key=lambda x: x["path"])


def collect_papers() -> list[dict]:
    """
    Collect all PDFs from rag/papers/ (recursively).
    """
    if not PAPERS_DIR.exists():
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        return []

    return sorted(
        [{"path": p, "source_type": "paper"}
         for p in PAPERS_DIR.rglob("*.pdf")],
        key=lambda x: x["path"],
    )


# ── 2. Extract text ─────────────────────────────────────────────────────────

def extract_pdf_text(path: Path) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz).

    PyMuPDF handles multi-column layouts and preserves reading order better
    than many alternatives.  It won't perfectly render LaTeX equations, but
    it captures the surrounding text, variable names, and equation numbers,
    which is usually enough for retrieval to find the right section.

    For papers with heavy math, consider also keeping the original PDF
    alongside and referencing equation numbers in your queries.
    """
    import fitz  # pymupdf

    doc = fitz.open(path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")  # plain text extraction
        if text.strip():
            pages.append(f"[Page {page_num + 1}]\n{text}")
    doc.close()
    return "\n\n".join(pages)


# ── 3. Chunk ─────────────────────────────────────────────────────────────────

# Different chunk sizes for different source types:
#   - Code: smaller chunks, language-aware (split at function/class boundaries)
#   - Papers: larger chunks with more overlap (preserve paragraph context)
#   - Docs: medium chunks

CHUNK_CONFIGS = {
    "code":  {"chunk_size": 1500, "chunk_overlap": 200},
    "docs":  {"chunk_size": 1500, "chunk_overlap": 200},
    "paper": {"chunk_size": 2000, "chunk_overlap": 400},  # larger for papers
}


def chunk_source(path: Path, source_type: str) -> list[dict]:
    """
    Split a file into overlapping text chunks with rich metadata.

    Each chunk carries:
      - source: relative path or paper filename
      - source_type: "code", "docs", or "paper"
      - language: file extension (py, rst, md, pdf)
      - chunk_index / total_chunks: position within the file
    """
    # Get the text
    if source_type == "paper":
        text = extract_pdf_text(path)
        rel_path = f"papers/{path.relative_to(PAPERS_DIR)}"
    else:
        text = path.read_text(errors="replace")
        rel_path = str(path.relative_to(REPO_ROOT))

    if not text.strip():
        return []

    # Choose splitter
    cfg = CHUNK_CONFIGS[source_type]
    if source_type == "code" and path.suffix == ".py":
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )
    else:
        # For papers, split on double-newlines (paragraph boundaries) first,
        # then fall back to single newlines, then sentences.
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )

    docs = splitter.create_documents(
        texts=[text],
        metadatas=[{
            "source":      rel_path,
            "source_type": source_type,
            "language":    path.suffix.lstrip("."),
        }],
    )

    chunks = []
    for i, doc in enumerate(docs):
        chunks.append({
            "id":       f"{rel_path}::chunk_{i}",
            "text":     doc.page_content,
            "metadata": {
                **doc.metadata,
                "chunk_index":  i,
                "total_chunks": len(docs),
            },
        })
    return chunks


# ── 4. Embed & store ─────────────────────────────────────────────────────────

def embed_and_store(all_chunks: list[dict], force: bool = False):
    """Embed all chunks and write them to ChromaDB."""

    if force and DB_DIR.exists():
        import shutil
        shutil.rmtree(DB_DIR)
        console.print("[yellow]Deleted existing vector store.[/yellow]")

    # Load embedding model
    console.print(f"Loading embedding model [cyan]{EMBEDDING_MODEL}[/cyan]...")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Embed
    console.print(f"Computing embeddings for {len(all_chunks)} chunks...")
    t0 = time.time()
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.embed_documents(texts)
    dt = time.time() - t0
    console.print(f"Embedded in [green]{dt:.1f}s[/green].")

    # Store
    console.print(f"Storing in [cyan]{DB_DIR}[/cyan]...")
    client = PersistentClient(path=str(DB_DIR))

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    BATCH = 500
    for start in range(0, len(all_chunks), BATCH):
        end = min(start + BATCH, len(all_chunks))
        batch = all_chunks[start:end]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embeddings[start:end],
            metadatas=[c["metadata"] for c in batch],
        )

    return collection


# ── 5. Main pipeline ─────────────────────────────────────────────────────────

def build_index(
    include_farms: bool = False,
    force: bool = False,
    papers_only: bool = False,
):
    all_chunks = []

    # ── Code & docs ──
    if not papers_only:
        sources = collect_code_and_docs(include_farms=include_farms)
        console.print(f"Found [bold]{len(sources)}[/bold] code/doc files.")
        for s in track(sources, description="Chunking code & docs..."):
            try:
                all_chunks.extend(chunk_source(s["path"], s["source_type"]))
            except Exception as e:
                console.print(f"[red]Skip {s['path']}: {e}[/red]")

    # ── Papers ──
    papers = collect_papers()
    if papers:
        console.print(f"Found [bold]{len(papers)}[/bold] PDF papers.")
        for s in track(papers, description="Chunking papers..."):
            try:
                chunks = chunk_source(s["path"], s["source_type"])
                all_chunks.extend(chunks)
                console.print(
                    f"  [dim]{s['path'].name}[/dim] → {len(chunks)} chunks"
                )
            except Exception as e:
                console.print(f"[red]Skip {s['path'].name}: {e}[/red]")
    else:
        console.print(
            "[yellow]No papers found.[/yellow] "
            "Drop PDFs into [cyan]rag/papers/[/cyan] and re-run."
        )

    if not all_chunks:
        console.print("[red]Nothing to index.[/red]")
        return

    console.print(f"\nTotal: [bold]{len(all_chunks)}[/bold] chunks.")

    # ── Summary by source type ──
    counts = {}
    for c in all_chunks:
        st = c["metadata"]["source_type"]
        counts[st] = counts.get(st, 0) + 1
    for st, n in sorted(counts.items()):
        console.print(f"  {st:>6}: {n} chunks")

    # ── Embed & store ──
    collection = embed_and_store(all_chunks, force=force)
    console.print(
        f"\n[bold green]Done![/bold green] "
        f"{collection.count()} chunks indexed into '{COLLECTION}'."
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index lilytorch code + research papers into a vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python rag/index.py                  # index everything\n"
            "  python rag/index.py --force           # fresh re-index\n"
            "  python rag/index.py --papers-only     # only re-index papers\n"
        ),
    )
    parser.add_argument("--force", action="store_true",
                        help="Delete existing index and rebuild from scratch")
    parser.add_argument("--papers-only", action="store_true",
                        help="Only index PDFs in rag/papers/")
    parser.add_argument("--include-farms", action="store_true",
                        help="Also index FARMS_V2 submodules (large)")
    args = parser.parse_args()

    build_index(
        include_farms=args.include_farms,
        force=args.force,
        papers_only=args.papers_only,
    )
