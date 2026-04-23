"""
rag/mcp_server.py — MCP server exposing lilytorch CFD retrieval to VS Code Copilot.

This turns the RAG pipeline into a set of tools that Copilot can call
directly during chat.  No Anthropic API key needed — this only does
local embedding + ChromaDB retrieval.  The LLM is Copilot itself.

TOOLS EXPOSED
─────────────
1. search_cfd_knowledge(query)
   → Searches EVERYTHING: code + docs + papers.
   → Also prepends the relevant section(s) of the knowledge base.

2. search_code(query)
   → Searches only .py source files.  Best for "how is X implemented?"

3. search_papers(query)
   → Searches only indexed PDFs.  Best for theory / equations.

4. get_cfd_reference(topic)
   → Returns specific sections from the permanent knowledge base
     (governing equations, BDIM, Poisson solvers, etc.)  No vector
     search — instant lookup by topic keyword.

STARTUP
───────
  cd /path/to/lilytorch
  python rag/mcp_server.py

Or via the VS Code MCP configuration (see .vscode/mcp.json).
"""

import sys
from pathlib import Path

# ── Lazy globals (loaded once on first tool call) ────────────────────────────
_embedder = None
_collection = None

DB_DIR          = Path(__file__).resolve().parent / "vectorstore"
COLLECTION_NAME = "lilytorch"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K           = 12

# ── Add rag/ to sys.path so we can import cfd_knowledge_base ─────────────────
_rag_dir = str(Path(__file__).resolve().parent)
if _rag_dir not in sys.path:
    sys.path.insert(0, _rag_dir)

from cfd_knowledge_base import KNOWLEDGE_BASE, EXPERT_SYSTEMS  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_loaded():
    """Lazily load the embedding model and vector store once."""
    global _embedder, _collection
    if _embedder is not None:
        return

    from chromadb import PersistentClient
    from langchain_community.embeddings import HuggingFaceEmbeddings

    if not DB_DIR.exists():
        raise RuntimeError(
            f"Vector store not found at {DB_DIR}. "
            "Run `python rag/index.py` first to build the index."
        )

    _embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    client = PersistentClient(path=str(DB_DIR))
    _collection = client.get_collection(COLLECTION_NAME)


def _retrieve(query: str, top_k: int = TOP_K, source_filter: str | None = None) -> str:
    """Embed query → search ChromaDB → return formatted string."""
    _ensure_loaded()
    query_embedding = _embedder.embed_query(query)
    where = {"source_type": source_filter} if source_filter else None

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    parts = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        source = meta.get("source", "unknown")
        stype  = meta.get("source_type", "?")
        sim    = 1 - dist
        label  = {"code": "CODE", "docs": "DOCS", "paper": "PAPER"}.get(stype, "?")
        parts.append(f"── [{label}] {source} (relevance: {sim:.2f}) ──\n{doc}")

    return "\n\n".join(parts) if parts else "No relevant results found."


# ── Knowledge base section lookup ────────────────────────────────────────────

# Parse the knowledge base into sections for fast topic lookup
_KB_SECTIONS: dict[str, str] = {}

def _parse_kb_sections():
    """Parse KNOWLEDGE_BASE into {section_title_lower: section_text}."""
    if _KB_SECTIONS:
        return
    current_title = ""
    current_lines = []
    for line in KNOWLEDGE_BASE.split("\n"):
        if line.startswith("## "):
            if current_title:
                _KB_SECTIONS[current_title] = "\n".join(current_lines)
            current_title = line[3:].strip().lower()
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_title:
        _KB_SECTIONS[current_title] = "\n".join(current_lines)

_parse_kb_sections()


def _lookup_kb(topic: str) -> str:
    """Find knowledge base sections matching the topic keywords."""
    topic_lower = topic.lower()
    matches = []
    for title, text in _KB_SECTIONS.items():
        if any(word in title for word in topic_lower.split()):
            matches.append(text)
    if not matches:
        # Fallback: search within section text
        for title, text in _KB_SECTIONS.items():
            if topic_lower in text.lower():
                matches.append(text)
                if len(matches) >= 3:
                    break
    return "\n\n".join(matches) if matches else f"No knowledge base section found for '{topic}'."


# ══════════════════════════════════════════════════════════════════════════════
# MCP Server definition
# ══════════════════════════════════════════════════════════════════════════════

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP(
    "lilytorch-cfd",
    instructions=(
        "lilytorch CFD expert tools. Use these to search the lilytorch codebase, "
        "indexed research papers, and a curated CFD knowledge base covering "
        "incompressible Navier-Stokes, BDIM2 immersed boundary method, Poisson "
        "solvers, advection schemes, and fluid-structure interaction with "
        "MuJoCo/FARMS articulated bodies."
    ),
)


@mcp.tool()
def search_cfd_knowledge(query: str) -> str:
    """Search ALL indexed sources (code, docs, and papers) for the query.

    Use this tool when you need to answer questions about the lilytorch
    CFD solver — its implementation, numerical methods, or related research.
    Returns the most relevant code snippets, documentation sections, and
    paper excerpts, plus matching knowledge base sections.

    Args:
        query: Natural language question about CFD, the solver, or related methods.
    """
    kb_context = _lookup_kb(query)
    retrieved = _retrieve(query, top_k=TOP_K)

    parts = []
    if kb_context and "No knowledge base section" not in kb_context:
        parts.append(f"## Knowledge Base (permanent reference)\n\n{kb_context}")
    parts.append(f"## Retrieved from indexed sources\n\n{retrieved}")
    return "\n\n---\n\n".join(parts)


@mcp.tool()
def search_code(query: str) -> str:
    """Search only the lilytorch Python source code.

    Use this when asked about specific implementations, function signatures,
    class hierarchies, or code patterns in the lilytorch solver.

    Args:
        query: What to search for in the source code.
    """
    return _retrieve(query, top_k=TOP_K, source_filter="code")


@mcp.tool()
def search_papers(query: str) -> str:
    """Search only the indexed research papers (PDFs).

    Use this for theoretical questions about numerical methods, BDIM,
    immersed boundary methods, multi-body dynamics, convergence analysis,
    or when the user references a specific paper.

    Args:
        query: Theory or paper-related question.
    """
    return _retrieve(query, top_k=TOP_K, source_filter="paper")


@mcp.tool()
def get_cfd_reference(topic: str) -> str:
    """Get specific sections from the CFD knowledge base by topic.

    Fast lookup (no vector search) into the permanent knowledge base covering:
    governing equations, pressure projection, BDIM/BDIM2, MAC grid, time
    integration, advection schemes, Poisson solvers, force computation,
    boundary conditions, variable density, FARMS coupling, PyTorch patterns,
    configuration, and stability.

    Args:
        topic: Topic keyword(s), e.g. "BDIM", "Poisson", "advection", "stability".
    """
    return _lookup_kb(topic)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mcp.run()
