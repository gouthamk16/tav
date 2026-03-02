# TAV: Topology-Aware Vector Routing

Hierarchical semantic search for PDFs. Extracts document structure (chapters → sections → paragraphs), embeds each level with topology-weighted vectors, and runs a 3-pass zoom search: chapters first, then sections within winners, then paragraphs.

No LLM needed for retrieval. Just structure + vectors.

## Install

```bash
pip install -e .
```

Optional backends:

```bash
pip install -e ".[postgres]"   # PostgreSQL + pgvector
pip install -e ".[mongo]"      # MongoDB
```

## Quick start

```bash
# Index a PDF
tav index --pdf_path document.pdf

# Query it
tav query --pdf_path document.pdf --query "how does memory management work"

# Index all PDFs under an S3 prefix (or set TAV_S3_PATH in .env)
tav index --s3_path s3://my-bucket/my-prefix
```

## Store backends

By default, indices are saved as local files. You can also use Postgres or MongoDB:

```bash
# PostgreSQL
tav index --pdf_path doc.pdf --store postgres --store_uri "postgresql://user:pass@host/db"

# MongoDB
tav index --pdf_path doc.pdf --store mongo --store_uri "mongodb://host:27017"

# Local file store output location (default: current working directory)
tav index --pdf_path doc.pdf --output_dir ./indexes
```

For S3 ingestion, configure AWS credentials/region in `.env` (or your environment) and optionally set:

- `TAV_S3_PATH=s3://bucket/prefix`
- `AWS_REGION=us-east-1`
- `AWS_PROFILE=default`
- `AWS_ENDPOINT_URL_S3=http://localhost:9000` (optional S3-compatible endpoint)

## Python API

```python
from tav import parse_pdf, build_index, semantic_zoom_search, retrieve_context

tree = parse_pdf("document.pdf")
data = build_index(tree)
results = semantic_zoom_search("your query", data)
ctx = retrieve_context(results, data["paragraph_meta"], data["section_meta"], data["chapter_meta"])
print(ctx["context"])
```

## Docs

See [docs/README.md](docs/README.md) for the full usage guide and architecture.
