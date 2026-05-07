"""
rag/query.py — CFD-expert assistant with persistent knowledge + RAG retrieval.

ARCHITECTURE
────────────
This uses a two-layer approach to make Claude a genuine CFD expert:

Layer 1: KNOWLEDGE BASE (persistent, cached)
   A comprehensive domain reference (~8k tokens) covering all the equations,
   methods, and terminology relevant to your solver.  This is sent as part of
   the system prompt and CACHED by the Anthropic API — meaning Claude "knows"
   all this for every query at 90% cost reduction after the first call.

Layer 2: RAG RETRIEVAL (per-query)
   On each question, we retrieve the most relevant chunks from your code,
   docs, and papers.  These are added to the user message so Claude can
   reference specific implementations and paper passages.

The result: Claude has permanent CFD expertise (layer 1) AND sees your specific
code/papers (layer 2) for every answer.

USAGE
─────
  python rag/query.py                                    # interactive
  python rag/query.py -q "How does BDIM handle BCs?"     # single question
  python rag/query.py --mode code -q "Write a SimConfig" # code generation
  python rag/query.py --mode theory --source paper -q .. # theory from papers

REQUIRED
────────
  export ANTHROPIC_API_KEY=sk-ant-...
  python rag/index.py   (must have run indexing first)
"""

import argparse
import os
import sys
from pathlib import Path

import anthropic
from chromadb import PersistentClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Import the knowledge base
from cfd_knowledge_base import KNOWLEDGE_BASE, EXPERT_SYSTEMS

# ── Configuration ─────────────────────────────────────────────────────────────

DB_DIR          = Path(__file__).resolve().parent / "vectorstore"
COLLECTION      = "lilytorch"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K           = 15
MODEL           = "claude-sonnet-4-20250514"

console = Console()


# ── Retrieval (same as before) ────────────────────────────────────────────────

def load_retriever():
    """Load the vector store and embedding model."""
    if not DB_DIR.exists():
        console.print(
            "[red]Vector store not found.[/red] "
            "Run [cyan]python rag/index.py[/cyan] first."
        )
        sys.exit(1)

    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    client = PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION)
    console.print(f"Loaded vector store: [bold]{collection.count()}[/bold] chunks.")
    return embedder, collection


def retrieve(
    query: str, embedder, collection,
    top_k: int = TOP_K,
    source_filter: str | None = None,
) -> tuple[str, list[dict]]:
    """Embed query → search ChromaDB → return formatted context."""
    query_embedding = embedder.embed_query(query)
    where = {"source_type": source_filter} if source_filter else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    context_parts = []
    raw_results = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        source = meta.get("source", "unknown")
        stype  = meta.get("source_type", "?")
        sim    = 1 - dist

        label = {"code": "CODE", "docs": "DOCS", "paper": "PAPER"}.get(stype, "?")
        context_parts.append(
            f"── [{label}] {source} (relevance: {sim:.2f}) ──\n{doc}"
        )
        raw_results.append({"source": source, "type": stype, "similarity": sim})

    return "\n\n".join(context_parts), raw_results


# ── Claude client with prompt caching ─────────────────────────────────────────

def create_client() -> anthropic.Anthropic:
    """Create the Anthropic API client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            "[red]ANTHROPIC_API_KEY not set.[/red]\n"
            "Run:  export ANTHROPIC_API_KEY=sk-ant-..."
        )
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def ask(
    query: str,
    context: str,
    client: anthropic.Anthropic,
    mode: str,
) -> tuple[str, dict]:
    """
    Build the two-layer prompt and send to Claude with prompt caching.

    PROMPT STRUCTURE
    ────────────────
    System message (CACHED — sent once, reused across queries):
    ┌─────────────────────────────────────────────────────────┐
    │  Block 1: Knowledge base (~8k tokens)     [cached]      │
    │  Block 2: Expert instructions for mode    [cached]      │
    └─────────────────────────────────────────────────────────┘

    User message (per-query):
    ┌─────────────────────────────────────────────────────────┐
    │  Retrieved code/paper chunks + the user's question      │
    └─────────────────────────────────────────────────────────┘

    WHY CACHING MATTERS
    ───────────────────
    The knowledge base is ~8k tokens.  Without caching, you'd pay for those
    tokens on every query.  With caching:
    - First query: full price (knowledge base gets cached server-side)
    - Subsequent queries: 90% discount on the cached portion
    - Cache lives for ~5 minutes of inactivity, then auto-refreshes
    """
    expert_instructions = EXPERT_SYSTEMS[mode]

    # Build system blocks with cache control
    # The cache_control marker tells Anthropic to cache everything up to here.
    system_blocks = [
        {
            "type": "text",
            "text": KNOWLEDGE_BASE,
        },
        {
            "type": "text",
            "text": expert_instructions,
            "cache_control": {"type": "ephemeral"},  # cache breakpoint
        },
    ]

    user_message = (
        f"## Retrieved context (code, docs, and/or papers)\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"## Question\n\n{query}"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        temperature=0.1,
        system=system_blocks,
        messages=[{"role": "user", "content": user_message}],
    )

    # Extract cache stats for reporting
    usage = response.usage
    cache_stats = {
        "input_tokens":        getattr(usage, "input_tokens", 0),
        "output_tokens":       getattr(usage, "output_tokens", 0),
        "cache_creation":      getattr(usage, "cache_creation_input_tokens", 0),
        "cache_read":          getattr(usage, "cache_read_input_tokens", 0),
    }

    answer = response.content[0].text
    return answer, cache_stats


# ── Interactive loop ──────────────────────────────────────────────────────────

def interactive(mode: str, source_filter: str | None, top_k: int):
    """Run an interactive chat session."""
    embedder, collection = load_retriever()
    client = create_client()

    filter_label = source_filter or "all"
    console.print(
        Panel(
            f"[bold]lilytorch CFD Expert[/bold]  "
            f"(mode: {mode}, sources: {filter_label})\n"
            f"Knowledge base loaded ({len(KNOWLEDGE_BASE):,} chars cached).\n"
            f"Retrieving top-{top_k} chunks per query.\n\n"
            "Type your question, or 'quit' to exit.\n"
            "Commands:  /mode chat|code|theory  /source all|code|paper|docs",
            border_style="blue",
        )
    )

    while True:
        try:
            raw = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw or raw.lower() in ("quit", "exit", "q"):
            break

        # In-session mode/source switching
        if raw.startswith("/mode "):
            new_mode = raw.split()[1]
            if new_mode in EXPERT_SYSTEMS:
                mode = new_mode
                console.print(f"[green]Switched to {mode} mode.[/green]")
            else:
                console.print("[red]Unknown mode. Use: chat, code, theory[/red]")
            continue
        if raw.startswith("/source "):
            new_src = raw.split()[1]
            source_filter = None if new_src == "all" else new_src
            console.print(f"[green]Source filter: {new_src}[/green]")
            continue

        # Retrieve
        with console.status("Retrieving relevant context..."):
            context, hits = retrieve(
                raw, embedder, collection,
                top_k=top_k, source_filter=source_filter,
            )

        types_found = {}
        for h in hits:
            types_found[h["type"]] = types_found.get(h["type"], 0) + 1
        summary = ", ".join(f"{n} {t}" for t, n in sorted(types_found.items()))
        console.print(f"[dim]Retrieved: {summary}[/dim]")

        # Generate
        with console.status("Generating answer..."):
            answer, cache_stats = ask(raw, context, client, mode)

        # Show cache status
        if cache_stats["cache_read"] > 0:
            console.print(
                f"[dim]Cache hit: {cache_stats['cache_read']} tokens read "
                f"from cache (90% cheaper)[/dim]"
            )
        elif cache_stats["cache_creation"] > 0:
            console.print(
                f"[dim]Cache created: {cache_stats['cache_creation']} tokens "
                f"cached for subsequent queries[/dim]"
            )

        console.print()
        console.print(Markdown(answer))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CFD-expert RAG assistant for lilytorch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python rag/query.py -q "How does BDIM handle BCs?"\n'
            '  python rag/query.py --mode code -q "Write a SimConfig for 3D sphere"\n'
            '  python rag/query.py --mode theory --source paper -q "BDIM convergence"\n'
        ),
    )
    parser.add_argument("-q", "--query", help="Single question (non-interactive)")
    parser.add_argument(
        "--mode", choices=["chat", "code", "theory"], default="chat",
        help="chat=Q&A, code=generate code, theory=deep explanations (default: chat)",
    )
    parser.add_argument(
        "--source", choices=["code", "paper", "docs"],
        default=None,
        help="Filter retrieval to one source type (default: all)",
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Number of chunks to retrieve (default: {TOP_K})",
    )
    args = parser.parse_args()

    if args.query:
        embedder, collection = load_retriever()
        client = create_client()
        context, hits = retrieve(
            args.query, embedder, collection,
            top_k=args.top_k, source_filter=args.source,
        )
        answer, cache_stats = ask(args.query, context, client, args.mode)
        console.print(Markdown(answer))

        # Print cache info
        if cache_stats["cache_read"] > 0:
            console.print(
                f"\n[dim]Cache hit: {cache_stats['cache_read']} tokens "
                f"from cache[/dim]"
            )
    else:
        interactive(args.mode, args.source, args.top_k)
