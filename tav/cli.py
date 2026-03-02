"""CLI for TAV: index and query commands. Returns raw structured context."""

import argparse
import json
import os
import sys
import tempfile
import time
from dotenv import load_dotenv

from .structural_parser import parse_pdf, print_tree
from .embedder import build_index, load_index
from .search import semantic_zoom_search
from .context_retriever import retrieve_context
from .store import get_store
from .s3_ingest import list_pdf_objects, download_pdf_objects, build_doc_name, parse_s3_uri

load_dotenv()


def _doc_name(pdf_path: str) -> str:
    return os.path.splitext(os.path.basename(pdf_path))[0]


def _index_dir(output_dir: str, doc_name: str) -> str:
    return os.path.join(output_dir, f".tav_index_{doc_name}")


def _resolve_doc_name(args, *, for_index: bool) -> str:
    if getattr(args, "doc_name", None):
        return args.doc_name
    if getattr(args, "pdf_path", None):
        return _doc_name(args.pdf_path)
    if for_index:
        s3_path = getattr(args, "s3_path", None) or os.environ.get("TAV_S3_PATH")
        if s3_path:
            bucket, prefix = parse_s3_uri(s3_path)
            return build_doc_name(bucket, prefix)
    return ""


def cmd_index(args):
    weights = tuple(float(w) for w in args.weights.split(","))
    if len(weights) != 3:
        print("Error: --weights must be 3 comma-separated floats (e.g., 0.7,0.2,0.1)")
        sys.exit(1)

    s3_path = args.s3_path or os.environ.get("TAV_S3_PATH")
    use_s3 = bool(s3_path)

    if not use_s3 and not args.pdf_path:
        print("Error: provide --pdf_path for local files, or --s3_path (or TAV_S3_PATH in .env) for S3 ingestion")
        sys.exit(1)

    if use_s3 and args.pdf_path:
        print("Error: use either --pdf_path or --s3_path, not both")
        sys.exit(1)

    name = _resolve_doc_name(args, for_index=True)
    if not name:
        print("Error: unable to resolve document name, pass --doc_name")
        sys.exit(1)

    if use_s3:
        print(f"Listing S3 PDFs: {s3_path}")
        t0 = time.time()
        try:
            objs = list_pdf_objects(
                s3_path,
                region_name=args.aws_region,
                profile=args.aws_profile,
                endpoint_url=args.s3_endpoint_url,
            )
        except Exception as e:
            print(f"Error: failed to list S3 objects: {e}")
            sys.exit(1)

        if not objs:
            print(f"Error: no PDF files found under {s3_path}")
            sys.exit(1)

        print(f"   Found {len(objs)} PDFs in {time.time() - t0:.2f}s")
        tree = []
        with tempfile.TemporaryDirectory(prefix="tav_s3_") as tmp_dir:
            print("Downloading PDFs...")
            try:
                downloaded = download_pdf_objects(
                    objs,
                    tmp_dir,
                    region_name=args.aws_region,
                    profile=args.aws_profile,
                    endpoint_url=args.s3_endpoint_url,
                )
            except Exception as e:
                print(f"Error: failed to download S3 PDFs: {e}")
                sys.exit(1)

            print("Parsing PDFs...")
            t0 = time.time()
            for item in downloaded:
                sub_tree = parse_pdf(item.local_path)
                prefix = f"[{item.key}] "
                for root in sub_tree:
                    root.title = prefix + root.title
                tree.extend(sub_tree)
            print(f"   Parsed in {time.time() - t0:.2f}s")
    else:
        if not os.path.isfile(args.pdf_path):
            print(f"Error: File not found: {args.pdf_path}")
            sys.exit(1)
        print(f"Parsing PDF: {args.pdf_path}")
        t0 = time.time()
        tree = parse_pdf(args.pdf_path)
        print(f"   Parsed in {time.time() - t0:.2f}s")

    print("\nDocument Tree:")
    print_tree(tree)

    store = _make_store(args)

    print(f"\nBuilding index (model: {args.embed_model})...")
    t0 = time.time()
    if args.store == "file":
        out = _index_dir(args.output_dir, name)
        data = build_index(tree, embed_model=args.embed_model, weights=weights, output_dir=out)
        label = out
    else:
        data = build_index(tree, embed_model=args.embed_model, weights=weights, store=store, doc_name=name)
        label = f"{args.store}:{name}"
    print(f"   Indexed in {time.time() - t0:.2f}s")

    cfg = data["config"]
    print(f"\nDone! Index saved to: {label}")
    print(f"   Chapters: {cfg['num_chapters']}, Sections: {cfg['num_sections']}, Paragraphs: {cfg['num_paragraphs']}")
    print(f"   Embedding dim: {cfg['dim']}")


def cmd_query(args):
    if not args.query:
        print("Error: --query is required")
        sys.exit(1)

    name = _resolve_doc_name(args, for_index=False)
    if not name:
        print("Error: provide --doc_name or --pdf_path")
        sys.exit(1)

    store = _make_store(args)
    use_store = args.store != "file"

    if use_store:
        try:
            data = store.load(name)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Run: tav index --pdf_path {args.pdf_path} --store {args.store} --store_uri <uri>")
            sys.exit(1)
        label = f"{args.store}:{name}"
    else:
        idx_dir = _index_dir(args.output_dir, name)
        if not os.path.isdir(idx_dir):
            print(f"Error: Index not found at {idx_dir}")
            if args.pdf_path:
                print(f"Run: tav index --pdf_path {args.pdf_path} --output_dir {args.output_dir}")
            else:
                print(f"Run: tav index --s3_path <s3://bucket/prefix> --doc_name {name} --output_dir {args.output_dir}")
            sys.exit(1)
        data = load_index(idx_dir)
        label = idx_dir

    model = data["config"]["embed_model"]

    print(f"Loading index from: {label}")
    print(f"Query: {args.query}")
    print(f"   Model: {model}")
    print(f"   Zoom: K_chapters={args.k_chapters}, K_sections={args.k_sections}, K_paragraphs={args.k_paragraphs}\n")

    t0 = time.time()
    if use_store:
        results = semantic_zoom_search(
            args.query, embed_model=model,
            k_chapters=args.k_chapters, k_sections=args.k_sections, k_paragraphs=args.k_paragraphs,
            store=store, doc_name=name,
        )
    else:
        results = semantic_zoom_search(
            args.query, data, embed_model=model,
            k_chapters=args.k_chapters, k_sections=args.k_sections, k_paragraphs=args.k_paragraphs,
        )
    elapsed = time.time() - t0

    print(f"Search completed in {elapsed * 1000:.1f}ms")
    print(f"   Found {len(results)} matching paragraphs\n")

    print("=" * 70)
    print("RETRIEVED NODES")
    print("=" * 70)
    for i, r in enumerate(results):
        print(f"\n[{i+1}] {r['hierarchy_path']}")
        print(f"    Pages: {r['page_start']}-{r['page_end']}  |  Score: {r['score']:.4f}")

    ctx = retrieve_context(
        results, data["paragraph_meta"], data["section_meta"], data["chapter_meta"],
        max_context_tokens=args.max_context_tokens,
    )

    print(f"\n{'=' * 70}")
    print(f"STRUCTURED CONTEXT ({ctx['token_count']} tokens)")
    print("=" * 70)
    print(ctx["context"])

    print(f"\n{'=' * 70}")
    print("SOURCES")
    print("=" * 70)
    for s in ctx["sources"]:
        print(f"   {s['hierarchy_path']}  (pp. {s['page_start']}-{s['page_end']})")

    if args.json_output:
        out = {
            "query": args.query,
            "results": [{k: v for k, v in r.items() if k != "text"} for r in results],
            "context": ctx["context"], "sources": ctx["sources"],
            "search_time_ms": round(elapsed * 1000, 1), "token_count": ctx["token_count"],
        }
        print(f"\n{json.dumps(out, indent=2, ensure_ascii=False)}")


def _make_store(args):
    backend = getattr(args, "store", "file")
    uri = getattr(args, "store_uri", None) or os.environ.get("TAV_STORE_URI")
    if backend == "file":
        base = getattr(args, "output_dir", ".")
        return get_store("file", base_dir=base)
    return get_store(backend, uri=uri)


def main():
    parser = argparse.ArgumentParser(prog="tav", description="TAV: Topology-Aware Vector Routing")
    sub = parser.add_subparsers(dest="command")

    # Shared store args
    store_args = argparse.ArgumentParser(add_help=False)
    store_args.add_argument("--store", default="file", choices=["file", "postgres", "mongo"],
                            help="Storage backend (default: file)")
    store_args.add_argument("--store_uri", default=None,
                            help="Connection URI for postgres/mongo (or set TAV_STORE_URI env var)")

    ix = sub.add_parser("index", help="Index a PDF", parents=[store_args])
    ix.add_argument("--pdf_path", default=None, help="Path to a local PDF file")
    ix.add_argument("--s3_path", default=None, help="S3 prefix path (s3://bucket/prefix). Can also set TAV_S3_PATH")
    ix.add_argument("--doc_name", default=None, help="Document name override")
    ix.add_argument("--output_dir", default=".", help="Output directory for local file store (default: current working directory)")
    ix.add_argument("--aws_region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
    ix.add_argument("--aws_profile", default=os.environ.get("AWS_PROFILE"))
    ix.add_argument("--s3_endpoint_url", default=os.environ.get("AWS_ENDPOINT_URL_S3"))
    ix.add_argument("--embed_model", default="all-MiniLM-L6-v2", help="'openai' or sentence-transformers model name")
    ix.add_argument("--weights", default="0.7,0.2,0.1", help="Topology weights: para,section,chapter")

    q = sub.add_parser("query", help="Query an indexed PDF", parents=[store_args])
    q.add_argument("--pdf_path", default=None, help="Path to source local PDF (optional if --doc_name provided)")
    q.add_argument("--doc_name", default=None, help="Indexed document name (required if no --pdf_path)")
    q.add_argument("--output_dir", default=".", help="Directory where local file indices are stored")
    q.add_argument("--query", required=True)
    q.add_argument("--k_chapters", type=int, default=3)
    q.add_argument("--k_sections", type=int, default=5)
    q.add_argument("--k_paragraphs", type=int, default=10)
    q.add_argument("--max_context_tokens", type=int, default=8000)
    q.add_argument("--json_output", action="store_true")

    args = parser.parse_args()
    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
