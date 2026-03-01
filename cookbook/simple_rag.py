"""
End-to-end RAG: TAV-RAG retrieval + OpenAI answer generation.

Usage:
    python -m cookbook.simple_rag --pdf_path doc.pdf --query "your question"

Requires OPENAI_API_KEY in .env and a pre-built index (run `tav index --pdf_path doc.pdf` first).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tav.embedder import load_index
from tav.search import semantic_zoom_search
from tav.context_retriever import retrieve_context
from dotenv import load_dotenv

load_dotenv()


def _index_dir(pdf_path):
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    return os.path.join(os.path.dirname(pdf_path), f".tav_index_{base}")


def generate_answer(query, context, model="gpt-4o-2024-11-20"):
    import openai
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env")

    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": (
            f"Answer the question based on the provided document context.\n"
            f"Cite section titles and page numbers.\n\n"
            f"Question: {query}\n\nContext:\n{context}"
        )}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="TAV + LLM answer generation")
    parser.add_argument("--pdf_path", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default="gpt-4o-2024-11-20")
    parser.add_argument("--k_chapters", type=int, default=3)
    parser.add_argument("--k_sections", type=int, default=5)
    parser.add_argument("--k_paragraphs", type=int, default=10)
    parser.add_argument("--max_context_tokens", type=int, default=8000)
    args = parser.parse_args()

    idx_dir = _index_dir(args.pdf_path)
    if not os.path.isdir(idx_dir):
        print(f" Index not found. Run: tav index --pdf_path {args.pdf_path}")
        sys.exit(1)

    data = load_index(idx_dir)
    model = data["config"]["embed_model"]

    print(f" Retrieving context for: {args.query}")
    t0 = time.time()
    results = semantic_zoom_search(args.query, data, embed_model=model,
                                    k_chapters=args.k_chapters, k_sections=args.k_sections, k_paragraphs=args.k_paragraphs)
    ctx = retrieve_context(results, data["paragraph_meta"], data["section_meta"], data["chapter_meta"],
                           max_context_tokens=args.max_context_tokens)
    ret_time = time.time() - t0
    print(f" Retrieved in {ret_time * 1000:.1f}ms ({ctx['token_count']} tokens)")

    print(f"\n Generating answer with {args.model}...")
    t0 = time.time()
    answer = generate_answer(args.query, ctx["context"], model=args.model)
    gen_time = time.time() - t0
    print(f" Generated in {gen_time:.2f}s\n")

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    print(f"\n{'=' * 70}")
    print("SOURCES")
    print("=" * 70)
    for s in ctx["sources"]:
        print(f"   {s['hierarchy_path']}  (pp. {s['page_start']}-{s['page_end']})")

    print(f"\n  Retrieval: {ret_time * 1000:.1f}ms  |  Generation: {gen_time:.2f}s")


if __name__ == "__main__":
    main()
