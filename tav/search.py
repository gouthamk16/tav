"""3-pass hierarchical vector routing: Chapter → Section → Paragraph. Pure math, no LLM."""

import numpy as np
import faiss
from typing import List, Dict, Optional

from .embedder import get_backend
from .store.base import IndexStore


def semantic_zoom_search(
    query: str,
    index_data: Dict = None,
    embed_model: str = "all-MiniLM-L6-v2",
    k_chapters: int = 3,
    k_sections: int = 5,
    k_paragraphs: int = 10,
    store: Optional[IndexStore] = None,
    doc_name: Optional[str] = None,
) -> List[Dict]:
    backend = get_backend(embed_model)
    query_vec = backend.embed([query])

    # If a store with search_vectors is provided, do DB-native search
    if store and doc_name:
        return _store_search(store, doc_name, query_vec, k_chapters, k_sections, k_paragraphs)

    # Fallback: in-memory FAISS (original path)
    chap_meta = index_data["chapter_meta"]
    sec_meta = index_data["section_meta"]
    para_meta = index_data["paragraph_meta"]

    # Level 1: chapters
    chap_ix = index_data["chapter_index"]
    if chap_ix.ntotal == 0:
        return []
    _, chap_ids = chap_ix.search(query_vec, min(k_chapters, chap_ix.ntotal))
    winning_chaps = {chap_meta[i]["chapter_idx"] for i in chap_ids[0] if i >= 0}

    # Level 2: sections within winning chapters
    sec_candidates = [i for i, s in enumerate(sec_meta) if s["chapter_idx"] in winning_chaps]
    if not sec_candidates:
        sec_candidates = list(range(len(sec_meta)))
    sec_results = _search_subset(query_vec, index_data["section_index"], sec_meta, sec_candidates, k_sections)
    winning_secs = {r["section_idx"] for r in sec_results}

    # Level 3: paragraphs within winning sections
    para_candidates = [i for i, p in enumerate(para_meta) if p["section_idx"] in winning_secs]
    if not para_candidates:
        para_candidates = list(range(len(para_meta)))
    para_results = _search_subset(query_vec, index_data["paragraph_index"], para_meta, para_candidates, k_paragraphs)

    results = []
    for pr in para_results:
        sec_idx, chap_idx = pr.get("section_idx", -1), pr.get("chapter_idx", -1)
        chap_title = chap_meta[chap_idx]["title"] if 0 <= chap_idx < len(chap_meta) else ""
        sec_title = sec_meta[sec_idx]["title"] if 0 <= sec_idx < len(sec_meta) else ""

        # Build hierarchy path, avoiding redundancy when para title starts with section title
        para_title = pr["title"]
        if para_title == sec_title or para_title.startswith(sec_title + " ("):
            path_parts = [chap_title, para_title]
        else:
            path_parts = [chap_title, sec_title, para_title]

        results.append({
            "node_id": pr["node_id"], "title": para_title,
            "page_start": pr["page_start"], "page_end": pr["page_end"],
            "text": pr.get("text", ""), "score": pr["score"],
            "chapter_title": chap_title, "section_title": sec_title,
            "hierarchy_path": " > ".join(filter(None, path_parts)),
        })

    return results


def _search_subset(query_vec, full_ix, meta, candidates, k):
    """Search a filtered subset by reconstructing candidate vectors into a temp index."""
    if not candidates:
        return []
    dim = full_ix.d
    k = min(k, len(candidates))
    vecs = np.zeros((len(candidates), dim), dtype=np.float32)
    for i, idx in enumerate(candidates):
        vecs[i] = full_ix.reconstruct(idx)
    tmp = faiss.IndexFlatIP(dim)
    tmp.add(vecs)
    scores, local_ids = tmp.search(query_vec, k)
    return [
        {**meta[candidates[lid]], "score": float(s)}
        for s, lid in zip(scores[0], local_ids[0]) if lid >= 0
    ]


def _store_search(store, doc_name, query_vec, k_chapters, k_sections, k_paragraphs):
    """3-pass zoom search delegated to a store backend."""
    # Level 1: chapters
    chap_results = store.search_vectors(doc_name, "chapter", query_vec, k_chapters)
    if not chap_results:
        return []
    winning_chaps = {r["chapter_idx"] for r in chap_results}

    # Level 2: sections — load section meta to find candidates, then search filtered
    data = store.load(doc_name)
    sec_meta = data["section_meta"]
    para_meta = data["paragraph_meta"]
    chap_meta = data["chapter_meta"]

    sec_candidates = [i for i, s in enumerate(sec_meta) if s["chapter_idx"] in winning_chaps]
    if not sec_candidates:
        sec_candidates = None  # no filter
    sec_results = store.search_vectors(doc_name, "section", query_vec, k_sections, filter_ids=sec_candidates)
    winning_secs = {r["section_idx"] for r in sec_results}

    # Level 3: paragraphs within winning sections
    para_candidates = [i for i, p in enumerate(para_meta) if p["section_idx"] in winning_secs]
    if not para_candidates:
        para_candidates = None
    para_results = store.search_vectors(doc_name, "paragraph", query_vec, k_paragraphs, filter_ids=para_candidates)

    results = []
    for pr in para_results:
        sec_idx = pr.get("section_idx", -1)
        chap_idx = pr.get("chapter_idx", -1)
        chap_title = chap_meta[chap_idx]["title"] if 0 <= chap_idx < len(chap_meta) else ""
        sec_title = sec_meta[sec_idx]["title"] if 0 <= sec_idx < len(sec_meta) else ""

        para_title = pr["title"]
        if para_title == sec_title or para_title.startswith(sec_title + " ("):
            path_parts = [chap_title, para_title]
        else:
            path_parts = [chap_title, sec_title, para_title]

        results.append({
            "node_id": pr["node_id"], "title": para_title,
            "page_start": pr["page_start"], "page_end": pr["page_end"],
            "text": pr.get("text", ""), "score": pr["score"],
            "chapter_title": chap_title, "section_title": sec_title,
            "hierarchy_path": " > ".join(filter(None, path_parts)),
        })
    return results
