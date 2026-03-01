"""TAV: Topology-Aware Vector Routing for documents."""

__version__ = "0.1.0"

from .structural_parser import parse_pdf, Node
from .embedder import build_index, load_index
from .search import semantic_zoom_search
from .context_retriever import retrieve_context
from .store import get_store
