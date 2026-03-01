"""Expands retrieved paragraphs into sibling-aware context blocks with token budgeting."""

import tiktoken
from typing import List, Dict


def retrieve_context(
    search_results: List[Dict],
    paragraph_meta: List[Dict],
    section_meta: List[Dict],
    chapter_meta: List[Dict],
    max_context_tokens: int = 8000,
    model: str = "gpt-4o-2024-11-20",
) -> Dict:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    # Dedupe and group by section
    seen = set()
    groups: Dict[int, List[Dict]] = {}
    for r in search_results:
        if r["node_id"] in seen:
            continue
        seen.add(r["node_id"])
        match = next((pm for pm in paragraph_meta if pm["node_id"] == r["node_id"]), None)
        if not match:
            continue
        sec_idx = match.get("section_idx", -1)
        groups.setdefault(sec_idx, []).append(match)

    blocks = []
    sources = []
    total_tokens = 0

    for sec_idx, matched in groups.items():
        sec_title = section_meta[sec_idx]["title"] if 0 <= sec_idx < len(section_meta) else ""
        chap_idx = section_meta[sec_idx].get("chapter_idx", -1) if 0 <= sec_idx < len(section_meta) else -1
        chap_title = chapter_meta[chap_idx]["title"] if 0 <= chap_idx < len(chapter_meta) else ""

        siblings = sorted(
            [pm for pm in paragraph_meta if pm.get("section_idx") == sec_idx],
            key=lambda x: x.get("paragraph_idx", 0),
        )

        matched_ids = set()
        for mp in matched:
            for i, sp in enumerate(siblings):
                if sp["node_id"] == mp["node_id"]:
                    matched_ids.add(i)

        # Expand to include immediate neighbors
        expanded = set()
        for idx in matched_ids:
            expanded |= {max(0, idx - 1), idx, min(len(siblings) - 1, idx + 1)}

        header = f"## {chap_title} > {sec_title}" if chap_title else f"## {sec_title}"
        parts = [header, ""]
        for i in sorted(expanded):
            text = siblings[i].get("text", "").strip()
            if text:
                parts.extend([text, ""])
        block = "\n".join(parts)

        tok_count = len(enc.encode(block))
        if total_tokens + tok_count > max_context_tokens:
            # Trim: only matched paragraphs, no siblings
            parts = [header, ""]
            for idx in sorted(matched_ids):
                text = siblings[idx].get("text", "").strip()
                if text:
                    parts.extend([text, ""])
            block = "\n".join(parts)
            tok_count = len(enc.encode(block))
            if total_tokens + tok_count > max_context_tokens:
                continue

        blocks.append(block)
        total_tokens += tok_count
        sources.append({
            "chapter": chap_title, "section": sec_title,
            "page_start": min(p["page_start"] for p in matched),
            "page_end": max(p["page_end"] for p in matched),
            "hierarchy_path": f"{chap_title} > {sec_title}" if chap_title else sec_title,
        })

    return {"context": "\n---\n\n".join(blocks), "sources": sources, "token_count": total_tokens}
