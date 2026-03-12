# Making Claude a CFD Expert — lilytorch RAG + Knowledge Base

## VS Code Copilot Integration (MCP)

**This is the main way to use it.** The RAG system runs as an MCP server
that Copilot calls automatically during chat. No extra API key needed —
Copilot IS the LLM.

### Setup (one-time)

```bash
# 1. Install dependencies
pip install -r rag/requirements.txt

# 2. Build the index
python rag/index.py

# 3. Done — the .vscode/mcp.json config is already in place
```

### How to use

Just chat with Copilot normally. The MCP server gives Copilot 4 tools:

| Tool | What it does | When Copilot uses it |
|------|-------------|---------------------|
| `search_cfd_knowledge` | Searches code + docs + papers + knowledge base | General CFD questions |
| `search_code` | Searches only .py files | "How is X implemented?" |
| `search_papers` | Searches only indexed PDFs | Theory / equations |
| `get_cfd_reference` | Instant lookup from knowledge base (no vector search) | Core method references |

You can also explicitly ask Copilot to use a tool:
> "Use search_papers to find what Weymouth & Yue say about the BDIM kernel"

### Cost

| Component | Cost |
|-----------|------|
| Copilot subscription | Whatever you already pay |
| MCP server | **Free** (runs locally) |
| Embeddings | **Free** (local model, ~90MB) |
| ChromaDB | **Free** (local) |
| Extra API calls | **None** |

**Total additional cost: $0.**

### Re-indexing after changes

```bash
# After adding papers or modifying code:
python rag/index.py --force

# After adding only papers:
python rag/index.py --papers-only
```

---

## What's in each file

| File | Role |
|------|------|
| `mcp_server.py` | **MCP server** — exposes 4 tools to VS Code Copilot |
| `cfd_knowledge_base.py` | **The core expertise** — equations, methods, terminology. Edit this to teach Copilot new things. |
| `index.py` | Indexes code + docs + PDF papers into ChromaDB |
| `query.py` | Standalone terminal query tool (uses Anthropic API directly — optional) |
| `papers/` | Drop your PDFs here |
| `requirements.txt` | Dependencies |

---

## How to teach Claude new things

### Option A: Add to the knowledge base (permanent knowledge)

Edit `cfd_knowledge_base.py` → `KNOWLEDGE_BASE` string. Add a new section:

```python
## 16. SMAGORINSKY LES MODEL

Subgrid-scale stress tensor:
$$\tau_{ij}^{sgs} = -2 (C_s \Delta)^2 |\bar{S}| \bar{S}_{ij}$$

where C_s ≈ 0.1-0.2 (Smagorinsky constant), Δ is filter width (= h),
|S̄| = sqrt(2 S̄_ij S̄_ij) is the strain rate magnitude.

This adds an effective turbulent viscosity:
$$\nu_t = (C_s \Delta)^2 |\bar{S}|$$

In lilytorch, this would modify the diffusion term in adv_diff.py...
```

**Use this for:** core methods, key equations, terminology, architectural patterns.

### Option B: Add a paper (retrieved per-query)

Drop a PDF in `rag/papers/` and re-index:

```bash
cp smagorinsky_1963.pdf rag/papers/les/
python rag/index.py --force
```

**Use this for:** detailed derivations, experimental data, specific results.

### Option C: Add a notes file (best for math)

PDF math extraction is imperfect. For critical equations, write them in
a markdown file:

```bash
cat > rag/papers/notes_les.md << 'EOF'
# LES Notes — Smagorinsky Model

## Key equations
- Subgrid stress: τ_ij^sgs = -2(C_s Δ)² |S̄| S̄_ij
- Effective viscosity: ν_t = (C_s Δ)² |S̄|
- C_s = 0.1 for channel flow (Deardorff 1970)
- C_s = 0.17 for isotropic turbulence (Lilly 1967)

## Implementation notes
- Add ν_t to ν in the diffusion term of adv_diff.py
- Compute strain rate from velocity gradients (operations.py)
- Need to compute |S̄| at cell centres, then interpolate to faces
EOF

python rag/index.py --force
```

**Use this for:** equations that extract poorly from PDFs, your own derivations.

---

## Standalone terminal mode (optional)

If you want to use this outside VS Code, query.py talks to the
Anthropic API directly with prompt caching:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python rag/query.py                                    # interactive
python rag/query.py -q "How does BDIM handle BCs?"     # single question
python rag/query.py --mode code -q "Write a SimConfig" # code generation
python rag/query.py --mode theory --source paper        # theory mode
```

This costs ~$0.003-0.01 per question (with prompt caching).

---

## Prompt caching — how it saves money

The knowledge base is ~8,000 tokens. Without caching, every query pays
for those tokens at full price.

With Anthropic's **prompt caching**:
- First query: full price. The knowledge base gets cached server-side.
- Subsequent queries (within ~5 min): 90% discount on cached tokens.
- The cache auto-refreshes on each use.

The terminal shows cache status:
```
Cache hit: 2847 tokens read from cache (90% cheaper)
```

**Cost estimate:** ~$0.003-0.01 per question. Heavy use (100 queries/day) ≈ $0.50-1.00.

---

## Three modes

| Mode | Use for | Example |
|------|---------|---------|
| `chat` | Understanding code, debugging, design decisions | "How does the multigrid V-cycle handle variable coefficients?" |
| `code` | Generating code that follows your conventions | "Write a SimConfig for a 3D falling sphere" |
| `theory` | Deep explanations with equations and paper citations | "Derive the BDIM meta-equation convergence properties" |

Switch modes in-session: `/mode theory`

## Source filtering

| Filter | What's searched | When to use |
|--------|----------------|-------------|
| (none) | Everything | Default — best for most questions |
| `--source paper` | Only indexed PDFs | Theory questions |
| `--source code` | Only .py files | Implementation questions |
| `--source docs` | Only .rst/.md | "What does the docs say about..." |

Switch in-session: `/source paper`

---

## Suggested papers to index

| Paper | Topic | Filename suggestion |
|-------|-------|-------------------|
| Weymouth & Yue (2011) | BDIM foundational paper | `bdim/weymouth_yue_2011.pdf` |
| Maertens & Weymouth (2015) | BDIM2 second-order | `bdim/maertens_weymouth_2015.pdf` |
| Mittal & Iaccarino (2005) | IBM review/taxonomy | `ibm/mittal_iaccarino_2005.pdf` |
| Leonard (1991) | ULTIMATE/ADBQUICKEST | `advection/leonard_1991.pdf` |
| Briggs, Henson & McCormick (2000) | Multigrid tutorial | `multigrid/briggs_2000.pdf` |
| Coquerelle & Cottet (2008) | Vortex penalisation | `fsi/coquerelle_cottet_2008.pdf` |
| Gazzola et al. (2011) | Optimised swimming | `fsi/gazzola_2011.pdf` |
| Peskin (2002) | IBM methods review | `ibm/peskin_2002.pdf` |

---

## Architecture

```
.vscode/
└── mcp.json               ← Tells Copilot to start the MCP server

rag/
├── mcp_server.py          ← MCP SERVER: 4 tools for Copilot
├── cfd_knowledge_base.py  ← THE CORE: all domain expertise lives here
├── index.py               ← Index code + docs + papers → ChromaDB
├── query.py               ← Standalone terminal tool (optional, uses Anthropic API)
├── requirements.txt
├── papers/                ← Your PDFs
│   ├── bdim/
│   ├── ibm/
│   ├── multigrid/
│   └── ...
└── vectorstore/           ← ChromaDB data (gitignored)
```
