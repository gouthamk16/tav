"""Topology-weighted embeddings with FAISS indices. Supports sentence-transformers (default) and OpenAI."""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

from .structural_parser import Node
from .store.base import IndexStore


class SentenceTransformerBackend:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array(
            self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True),
            dtype=np.float32,
        )


class OpenAIBackend:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required in .env for OpenAI backend.")
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.dim = 1536

    def embed(self, texts: List[str]) -> np.ndarray:
        all_vecs = []
        for i in range(0, len(texts), 100):
            resp = self.client.embeddings.create(model=self.model_name, input=texts[i:i+100])
            all_vecs.extend(item.embedding for item in resp.data)
        arr = np.array(all_vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return arr / norms


def get_backend(embed_model: str = "all-MiniLM-L6-v2"):
    if embed_model.lower() == "openai":
        return OpenAIBackend()
    return SentenceTransformerBackend(model_name=embed_model)


def _node_text(node: Node, max_chars: int = 500) -> str:
    parts = [node.title]
    if node.children:
        parts.append("Subtopics: " + ", ".join(c.title for c in node.children))
    if node.text:
        preview = node.text[:max_chars].strip()
        if preview:
            parts.append(preview)
    return " — ".join(parts)


def _apply_topology_weights(para_vecs, sec_indices, chap_indices, sec_vecs, chap_vecs, weights):
    """Blend paragraph vectors with parent section/chapter vectors, then L2-normalize."""
    w_p, w_s, w_c = weights
    out = np.zeros_like(para_vecs)
    for i in range(len(para_vecs)):
        v = w_p * para_vecs[i]
        si = sec_indices[i]
        if 0 <= si < len(sec_vecs):
            v += w_s * sec_vecs[si]
        ci = chap_indices[i]
        if 0 <= ci < len(chap_vecs):
            v += w_c * chap_vecs[ci]
        out[i] = v
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return out / norms


def build_index(
    tree: List[Node],
    embed_model: str = "all-MiniLM-L6-v2",
    weights: Tuple[float, float, float] = (0.70, 0.20, 0.10),
    output_dir: str = None,
    store: Optional[IndexStore] = None,
    doc_name: Optional[str] = None,
) -> Dict:
    backend = get_backend(embed_model)
    dim = backend.dim

    chapters, sections, paragraphs = [], [], []
    chap_idx = sec_idx = 0

    for root in tree:
        cur_chap = chap_idx
        chapters.append(root)
        chap_idx += 1

        if root.children:
            for sec_node in root.children:
                cur_sec = sec_idx
                sections.append((sec_node, cur_chap))
                sec_idx += 1
                if sec_node.children:
                    for para in sec_node.children:
                        paragraphs.append((para, cur_sec, cur_chap))
                else:
                    # Single-page leaf section — use it directly as paragraph
                    paragraphs.append((sec_node, cur_sec, cur_chap))
        else:
            # Childless root — treat as both section and paragraph
            sections.append((root, cur_chap))
            paragraphs.append((root, sec_idx, cur_chap))
            sec_idx += 1

    print(f"Embedding {len(chapters)} chapters, {len(sections)} sections, {len(paragraphs)} paragraphs...")

    empty = np.zeros((0, dim), dtype=np.float32)
    chap_vecs = backend.embed([_node_text(n) for n in chapters]) if chapters else empty
    sec_vecs = backend.embed([_node_text(s[0]) for s in sections]) if sections else empty
    raw_para = backend.embed([_node_text(p[0]) for p in paragraphs]) if paragraphs else empty

    if len(raw_para) > 0:
        para_vecs = _apply_topology_weights(
            raw_para, [p[1] for p in paragraphs], [p[2] for p in paragraphs],
            sec_vecs, chap_vecs, weights,
        )
    else:
        para_vecs = raw_para

    chap_ix, sec_ix, para_ix = faiss.IndexFlatIP(dim), faiss.IndexFlatIP(dim), faiss.IndexFlatIP(dim)
    if len(chap_vecs): chap_ix.add(chap_vecs)
    if len(sec_vecs): sec_ix.add(sec_vecs)
    if len(para_vecs): para_ix.add(para_vecs)

    chap_meta = [
        {"node_id": n.node_id, "title": n.title, "page_start": n.page_start,
         "page_end": n.page_end, "level": n.level, "chapter_idx": i}
        for i, n in enumerate(chapters)
    ]
    sec_meta = [
        {"node_id": s[0].node_id, "title": s[0].title, "page_start": s[0].page_start,
         "page_end": s[0].page_end, "level": s[0].level, "chapter_idx": s[1], "section_idx": i}
        for i, s in enumerate(sections)
    ]
    para_meta = [
        {"node_id": p[0].node_id, "title": p[0].title, "page_start": p[0].page_start,
         "page_end": p[0].page_end, "level": p[0].level, "section_idx": p[1],
         "chapter_idx": p[2], "paragraph_idx": i, "text": p[0].text}
        for i, p in enumerate(paragraphs)
    ]

    cfg = {
        "embed_model": embed_model, "dim": dim, "weights": list(weights),
        "num_chapters": len(chapters), "num_sections": len(sections), "num_paragraphs": len(paragraphs),
    }

    res = {
        "chapter_index": chap_ix, "section_index": sec_ix, "paragraph_index": para_ix,
        "chapter_meta": chap_meta, "section_meta": sec_meta, "paragraph_meta": para_meta,
        "config": cfg,
    }

    if store and doc_name:
        store.save(doc_name, res)
        print(f"Index saved to store ({type(store).__name__}: {doc_name})")
    elif output_dir:
        _save(res, output_dir)
        print(f"Index saved to {output_dir}")

    return res


def _save(data: Dict, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    for name in ("chapter", "section", "paragraph"):
        faiss.write_index(data[f"{name}_index"], os.path.join(path, f"{name}.faiss"))
        with open(os.path.join(path, f"{name}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(data[f"{name}_meta"], f, indent=2, ensure_ascii=False)
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(data["config"], f, indent=2, ensure_ascii=False)


def load_index(index_dir: str) -> Dict:
    with open(os.path.join(index_dir, "config.json")) as f:
        cfg = json.load(f)
    res = {"config": cfg}
    for name in ("chapter", "section", "paragraph"):
        res[f"{name}_index"] = faiss.read_index(os.path.join(index_dir, f"{name}.faiss"))
        with open(os.path.join(index_dir, f"{name}_meta.json"), encoding="utf-8") as f:
            res[f"{name}_meta"] = json.load(f)
    return res
