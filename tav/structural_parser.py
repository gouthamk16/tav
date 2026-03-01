"""Deterministic PDF hierarchy extraction via PyMuPDF TOC + font-size heuristics."""

import re
import pymupdf
from dataclasses import dataclass, field
from typing import List
from collections import Counter


@dataclass
class Node:
    node_id: str = ""
    level: int = 1
    title: str = ""
    page_start: int = 0
    page_end: int = 0
    text: str = ""
    children: List["Node"] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "node_id": self.node_id, "level": self.level, "title": self.title,
            "page_start": self.page_start, "page_end": self.page_end,
        }
        if self.text:
            d["text"] = self.text
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


def _clean_text(text: str) -> str:
    """Fix common PDF extraction artifacts."""
    # Fix hyphenation at line breaks: "com-\nputer" → "computer"
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = " ".join(line.split())  # normalize internal whitespace
        if not line:
            continue
        # Skip standalone page numbers
        if re.fullmatch(r'\d{1,4}', line.strip()):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _get_page_text(doc, page_num_1indexed: int) -> str:
    """Extract and clean text for a single page (1-indexed)."""
    idx = page_num_1indexed - 1
    if idx < 0 or idx >= len(doc):
        return ""
    raw = doc[idx].get_text("text") or ""
    return _clean_text(raw)


def _get_range_text(doc, start: int, end: int) -> str:
    parts = []
    for p in range(start - 1, min(end, len(doc))):
        t = doc[p].get_text("text")
        if t:
            parts.append(t)
    return _clean_text("\n".join(parts))


def _assign_ids(nodes: List[Node]) -> None:
    queue = []
    def collect(ns):
        for n in ns:
            queue.append(n)
            collect(n.children)
    collect(nodes)
    for i, node in enumerate(queue):
        node.node_id = f"{i:04d}"


def _propagate_page_end(nodes: List[Node]) -> None:
    """Bottom-up fix: parent page_end = max of its children's page_end."""
    for node in nodes:
        if node.children:
            _propagate_page_end(node.children)
            child_max = max(c.page_end for c in node.children)
            node.page_end = max(node.page_end, child_max)


def _expand_leaves(nodes: List[Node], doc) -> None:
    """For leaf nodes spanning >1 page, add per-page children with real PDF text."""
    for node in nodes:
        if node.children:
            _expand_leaves(node.children, doc)
            continue

        span = node.page_end - node.page_start + 1
        if span <= 1:
            continue

        for pg in range(node.page_start, node.page_end + 1):
            text = _get_page_text(doc, pg)
            if not text.strip():
                continue
            node.children.append(Node(
                level=node.level + 1,
                title=f"{node.title} (p. {pg})",
                page_start=pg,
                page_end=pg,
                text=text,
            ))


def _build_tree(flat_nodes: List[Node]) -> List[Node]:
    """Stack-based nesting of flat nodes using their level field."""
    if not flat_nodes:
        return []
    roots: List[Node] = []
    stack: List[Node] = []
    for node in flat_nodes:
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)
    return roots


def _parse_from_toc(doc: pymupdf.Document) -> Optional[List[Node]]:
    """Extract hierarchy from PDF bookmarks. Returns None if no usable TOC."""
    toc = doc.get_toc(simple=True)
    if not toc or len(toc) < 2:
        return None

    total = len(doc)
    flat = []
    for level, title, page in toc:
        flat.append(Node(level=level, title=title.strip(), page_start=max(1, page), page_end=total))

    for i in range(len(flat) - 1):
        nxt = flat[i + 1].page_start
        flat[i].page_end = max(flat[i].page_start, nxt - 1 if nxt > flat[i].page_start else nxt)
    flat[-1].page_end = total

    for node in flat:
        node.text = _get_range_text(doc, node.page_start, node.page_end)

    return _build_tree(flat)


def _parse_from_fonts(doc: pymupdf.Document) -> List[Node]:
    """Fallback: classify headings by font size when no TOC exists."""
    total = len(doc)
    spans = []
    for page_idx in range(total):
        blocks = doc[page_idx].get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text and len(text) > 1:
                        size = round(span.get("size", 12), 1)
                        spans.append((page_idx, size, text))

    if not spans:
        return _one_node_per_page(doc)

    size_counts = Counter(s[1] for s in spans)
    body_size = max(size_counts, key=size_counts.get)
    heading_sizes = sorted([s for s in size_counts if s > body_size], reverse=True)

    if not heading_sizes:
        return _one_node_per_page(doc)

    size_to_level = {sz: i + 1 for i, sz in enumerate(heading_sizes[:3])}
    flat: List[Node] = []
    body_lines: List[str] = []

    for page_idx, font_size, text in spans:
        if font_size in size_to_level:
            if flat and body_lines:
                flat[-1].text = _clean_text("\n".join(body_lines))
                body_lines = []
            flat.append(Node(level=size_to_level[font_size], title=text, page_start=page_idx + 1, page_end=total))
        else:
            body_lines.append(text)

    if flat and body_lines:
        flat[-1].text = _clean_text("\n".join(body_lines))

    if not flat:
        return _one_node_per_page(doc)

    for i in range(len(flat) - 1):
        flat[i].page_end = max(flat[i].page_start, flat[i + 1].page_start - 1)
    flat[-1].page_end = total

    for node in flat:
        if not node.text:
            node.text = _get_range_text(doc, node.page_start, node.page_end)

    return _build_tree(flat)


def _one_node_per_page(doc: pymupdf.Document) -> List[Node]:
    return [
        Node(level=1, title=f"Page {i+1}", page_start=i+1, page_end=i+1, text=_get_page_text(doc, i+1))
        for i in range(len(doc))
    ]


def _remove_noise_nodes(nodes: List[Node]) -> List[Node]:
    noise = {"contents", "table of contents", "index", "appendixes", "appendices", "bibliography", "references"}
    filtered = []
    for node in nodes:
        if node.title.strip().lower() in noise and not node.children:
            continue
        node.children = _remove_noise_nodes(node.children)
        filtered.append(node)
    return filtered


def parse_pdf(pdf_path: str) -> List[Node]:
    """Parse a PDF into a hierarchy. Tries native TOC first, falls back to font heuristics."""
    doc = pymupdf.open(pdf_path)

    tree = _parse_from_toc(doc)
    if not tree:
        tree = _parse_from_fonts(doc)

    tree = _remove_noise_nodes(tree)
    _propagate_page_end(tree)
    _expand_leaves(tree, doc)
    _assign_ids(tree)

    doc.close()
    return tree


def get_all_nodes_flat(tree: List[Node]) -> List[Node]:
    res = []
    def walk(ns):
        for n in ns:
            res.append(n)
            walk(n.children)
    walk(tree)
    return res


def print_tree(tree: List[Node], indent: int = 0) -> None:
    for node in tree:
        prefix = "  " * indent
        children_info = f" [{len(node.children)} children]" if node.children else ""
        print(f"{prefix}[{node.node_id}] L{node.level}: {node.title} (pp. {node.page_start}-{node.page_end}){children_info}")
        if node.children:
            print_tree(node.children, indent + 1)
