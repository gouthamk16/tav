# TAV — Usage Guide & Architecture

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Storage Backends](#storage-backends)
- [Architecture](#architecture)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Embedding Models](#embedding-models)
- [Topology Weights](#topology-weights)
- [Streamlit Explorer](#streamlit-explorer)
- [Cookbook: RAG with LLM](#cookbook-rag-with-llm)

---

## Overview

TAV (Topology-Aware Vector Routing) is a hierarchical semantic search engine for PDF documents. Instead of splitting documents into flat chunks, TAV:

1. **Extracts document structure** — chapters, sections, paragraphs — from the PDF's table of contents and font-size heuristics.
2. **Embeds each level** with topology-weighted vectors that blend a node's own embedding with its parent's, so paragraphs carry structural context.
3. **Searches via 3-pass zoom** — narrows from chapters to sections to paragraphs, so retrieval follows the document's own organization.

The result is structured context with hierarchy paths, page numbers, and token budgeting — ready to feed into an LLM or use standalone.

---

## Installation

Requires Python 3.10+.

```bash
# Core install
pip install -e .

# With PostgreSQL support
pip install -e ".[postgres]"

# With MongoDB support
pip install -e ".[mongo]"

# With Streamlit UI
pip install -e ".[app]"

# Everything
pip install -e ".[postgres,mongo,app]"
```

For OpenAI embeddings, set `OPENAI_API_KEY` in a `.env` file at the project root.

---

## CLI Reference

TAV installs as a `tav` command with two subcommands.

### `tav index`

Parse a PDF and build the vector index.

```bash
tav index --pdf_path <path> [options]
```

| Flag            | Default            | Description                                        |
| --------------- | ------------------ | -------------------------------------------------- |
| `--pdf_path`    | (required)         | Path to the PDF file                               |
| `--embed_model` | `all-MiniLM-L6-v2` | `"openai"` or any sentence-transformers model name |
| `--weights`     | `0.7,0.2,0.1`      | Topology weights: paragraph, section, chapter      |
| `--store`       | `file`             | Storage backend: `file`, `postgres`, `mongo`       |
| `--store_uri`   | `None`             | Connection URI (or set `TAV_STORE_URI` env var)    |

Examples:

```bash
# Local file store (default)
tav index --pdf_path textbook.pdf

# OpenAI embeddings
tav index --pdf_path textbook.pdf --embed_model openai

# PostgreSQL backend
tav index --pdf_path textbook.pdf --store postgres --store_uri "postgresql://user:pass@localhost/tav"

# MongoDB backend
tav index --pdf_path textbook.pdf --store mongo --store_uri "mongodb://localhost:27017"

# Custom topology weights (heavier section influence)
tav index --pdf_path textbook.pdf --weights 0.6,0.3,0.1
```

### `tav query`

Search an indexed PDF.

```bash
tav query --pdf_path <path> --query <text> [options]
```

| Flag                   | Default    | Description                        |
| ---------------------- | ---------- | ---------------------------------- |
| `--pdf_path`           | (required) | Path to the original PDF           |
| `--query`              | (required) | Search query text                  |
| `--k_chapters`         | `3`        | Top-K chapters in level 1          |
| `--k_sections`         | `5`        | Top-K sections in level 2          |
| `--k_paragraphs`       | `10`       | Top-K paragraphs in level 3        |
| `--max_context_tokens` | `8000`     | Token budget for assembled context |
| `--json_output`        | `false`    | Print machine-readable JSON output |
| `--store`              | `file`     | Storage backend                    |
| `--store_uri`          | `None`     | Connection URI                     |

Examples:

```bash
# Basic query
tav query --pdf_path textbook.pdf --query "how does garbage collection work"

# Narrow search, JSON output
tav query --pdf_path textbook.pdf --query "page tables" \
    --k_chapters 2 --k_sections 3 --k_paragraphs 5 --json_output

# Query from PostgreSQL
tav query --pdf_path textbook.pdf --query "virtual memory" \
    --store postgres --store_uri "postgresql://user:pass@localhost/tav"
```

---

## Python API

### Core pipeline

```python
from tav import parse_pdf, build_index, semantic_zoom_search, retrieve_context

# 1. Parse PDF into a hierarchy tree
tree = parse_pdf("document.pdf")

# 2. Build index (returns dict with FAISS indices + metadata)
data = build_index(tree, embed_model="all-MiniLM-L6-v2", weights=(0.7, 0.2, 0.1))

# 3. Search
results = semantic_zoom_search(
    "your query", data,
    k_chapters=3, k_sections=5, k_paragraphs=10,
)

# 4. Assemble context with token budgeting
ctx = retrieve_context(
    results,
    data["paragraph_meta"],
    data["section_meta"],
    data["chapter_meta"],
    max_context_tokens=8000,
)

print(ctx["context"])    # Structured text with headers
print(ctx["sources"])    # List of {chapter, section, page_start, page_end, hierarchy_path}
print(ctx["token_count"])
```

### With a store backend

```python
from tav import parse_pdf, build_index, semantic_zoom_search, get_store

store = get_store("postgres", uri="postgresql://user:pass@localhost/tav")

# Index
tree = parse_pdf("document.pdf")
build_index(tree, store=store, doc_name="document")

# Query (search runs in-DB via pgvector, no FAISS needed)
results = semantic_zoom_search(
    "your query",
    embed_model="all-MiniLM-L6-v2",
    store=store,
    doc_name="document",
)

# Load full index data if needed for context assembly
data = store.load("document")
```

### Store operations

```python
store = get_store("mongo", uri="mongodb://localhost:27017")

store.save("mybook", index_data)        # Persist
data = store.load("mybook")             # Load
store.delete("mybook")                  # Remove
docs = store.list_documents()           # List all indexed documents
results = store.search_vectors(         # Direct vector search
    "mybook", "paragraph", query_vec, k=10, filter_ids=[0, 1, 5]
)
```

---

## Storage Backends

### File (default)

Stores FAISS binary files + JSON metadata alongside the PDF.

```
/path/to/document.pdf
/path/to/.tav_index_document/
    chapter.faiss
    section.faiss
    paragraph.faiss
    chapter_meta.json
    section_meta.json
    paragraph_meta.json
    config.json
```

No extra dependencies. Good for local/single-user use.

### PostgreSQL + pgvector

Stores everything in two tables: `tav_documents` and `tav_nodes`. Vector search runs in-DB via pgvector's inner product operator (`<#>`), so there's no FAISS dependency at query time.

Prerequisites:

- PostgreSQL with the `vector` extension installed
- `pip install -e ".[postgres]"`

The schema is auto-created on first connection:

```sql
tav_documents (id UUID, name TEXT UNIQUE, embed_model, dim, weights, created_at)
tav_nodes     (id UUID, document_id FK, kind, node_id, idx, chapter_idx, section_idx,
               title, page_start, page_end, level, body, embedding vector)
```

### MongoDB

Stores documents in `documents` and `nodes` collections. Two vector search modes:

1. **Atlas Vector Search** — if you have an Atlas cluster with a vector search index named `vector_index` on the `nodes` collection, queries use `$vectorSearch` aggregation. Zero FAISS at query time.
2. **FAISS fallback** — for self-hosted Mongo, vectors are pulled from the DB and searched via a temp FAISS index. Still eliminates file-system coupling.

Prerequisites:

- `pip install -e ".[mongo]"`

```python
# Force FAISS fallback (no Atlas)
store = get_store("mongo", uri="mongodb://localhost:27017", atlas_vector_index=None)
```

---

## Architecture

```
tav/
├── structural_parser.py   # PDF → hierarchy tree (Node dataclass)
├── embedder.py            # Tree → topology-weighted embeddings + FAISS indices
├── search.py              # 3-pass semantic zoom search
├── context_retriever.py   # Expand results into token-budgeted context blocks
├── cli.py                 # CLI entry point
├── store/
│   ├── base.py            # IndexStore protocol
│   ├── file_store.py      # FAISS + JSON on disk
│   ├── pg_store.py        # PostgreSQL + pgvector
│   └── mongo_store.py     # MongoDB (Atlas or FAISS fallback)
├── __init__.py            # Public API exports
└── __main__.py            # python -m tav
```

### Data flow

```
PDF
 │
 ▼
structural_parser.parse_pdf()
 │  Extracts TOC + font-size heuristics → List[Node] tree
 │  Node: {node_id, level, title, page_start, page_end, text, children}
 │
 ▼
embedder.build_index()
 │  1. Flatten tree into chapters / sections / paragraphs
 │  2. Embed each level (sentence-transformers or OpenAI)
 │  3. Apply topology weights: para_vec = 0.7·para + 0.2·section + 0.1·chapter
 │  4. L2-normalize blended vectors
 │  5. Build FAISS IndexFlatIP per level
 │  6. Save via store or to disk
 │
 ▼
search.semantic_zoom_search()
 │  Pass 1: query → top-K chapters (inner product)
 │  Pass 2: query → top-K sections WITHIN winning chapters
 │  Pass 3: query → top-K paragraphs WITHIN winning sections
 │  Returns: [{node_id, title, page_start, page_end, text, score, hierarchy_path}]
 │
 ▼
context_retriever.retrieve_context()
    1. Deduplicate and group results by section
    2. Expand to neighboring sibling paragraphs for coherence
    3. Assemble text blocks with hierarchy headers
    4. Token-budget: if over limit, drop siblings, then drop blocks
    Returns: {context: str, sources: list, token_count: int}
```

### Topology weighting

Standard RAG chunks lose structural context. A paragraph about "page tables" in a "Virtual Memory" chapter gets the same treatment as one in "File Systems". TAV fixes this by blending parent vectors:

```
final_paragraph_vector = normalize(
    0.70 * paragraph_embedding
  + 0.20 * parent_section_embedding
  + 0.10 * parent_chapter_embedding
)
```

This means two paragraphs with identical text but different structural parents produce different vectors, biasing search toward structurally relevant results.

### Semantic zoom search

Instead of flat top-K across all chunks:

1. **Chapter pass** — find the 3 most relevant chapters. This narrows the search space.
2. **Section pass** — within those chapters, find the 5 most relevant sections.
3. **Paragraph pass** — within those sections, find the 10 most relevant paragraphs.

This mirrors how a human would navigate: table of contents first, then skim sections, then read paragraphs. It's faster (searches smaller subsets) and more precise (structural filtering removes false positives from unrelated chapters).

### Context retriever

Retrieved paragraphs are expanded into coherent blocks:

- Sibling paragraphs (immediately before/after in the same section) are included for context.
- Blocks are headed with `## Chapter > Section` for LLM grounding.
- A token budget (default 8000) controls total output size — if a block is too large, siblings are dropped first, then the block is skipped entirely.

---

## Embedding Models

TAV supports two embedding backends:

| Backend               | CLI flag                                   | Dim  | Notes                                                 |
| --------------------- | ------------------------------------------ | ---- | ----------------------------------------------------- |
| sentence-transformers | `--embed_model all-MiniLM-L6-v2` (default) | 384  | Local, no API key, fast                               |
| OpenAI                | `--embed_model openai`                     | 1536 | Uses `text-embedding-3-small`, needs `OPENAI_API_KEY` |

Any model from the [sentence-transformers hub](https://huggingface.co/models?library=sentence-transformers) works — just pass its name:

```bash
tav index --pdf_path doc.pdf --embed_model all-mpnet-base-v2
```

---

## Topology Weights

The `--weights` flag controls how much of each parent level bleeds into paragraph vectors.

```
--weights <paragraph>,<section>,<chapter>
```

| Preset          | Weights       | Use case                                         |
| --------------- | ------------- | ------------------------------------------------ |
| Default         | `0.7,0.2,0.1` | Balanced — paragraph content dominates           |
| Structure-heavy | `0.5,0.3,0.2` | When section/chapter context matters a lot       |
| Flat            | `1.0,0.0,0.0` | No topology — equivalent to standard chunked RAG |

Weights must sum to 1.0 (not enforced, but recommended). Vectors are L2-normalized after blending.

---

## Streamlit Explorer

A visual UI for parsing, viewing the document tree, building indices, and running searches.

```bash
pip install -e ".[app]"
streamlit run app.py
```

Upload a PDF, inspect the hierarchy, build an index, and search — all in the browser.

---

## Cookbook: RAG with LLM

`cookbook/simple_rag.py` shows a full pipeline: TAV retrieval + OpenAI answer generation.

```bash
# Index first
tav index --pdf_path textbook.pdf

# Ask a question
python -m cookbook.simple_rag --pdf_path textbook.pdf --query "What is virtual memory?"
```

This retrieves context via TAV, then feeds it to GPT-4o with a citation prompt. Requires `OPENAI_API_KEY`.
