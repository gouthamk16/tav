"""Abstract store protocol for index persistence."""

from __future__ import annotations
from typing import Protocol, runtime_checkable, Optional
import numpy as np


@runtime_checkable
class IndexStore(Protocol):
    """Backend-agnostic interface for storing and querying TAV indices."""

    def save(self, name: str, index_data: dict) -> str:
        """Persist index_data for a document. Returns a document identifier."""
        ...

    def load(self, name: str) -> dict:
        """Load index_data dict (same shape as build_index output)."""
        ...

    def delete(self, name: str) -> None:
        ...

    def list_documents(self) -> list[dict]:
        ...

    def search_vectors(
        self,
        name: str,
        kind: str,
        query_vec: np.ndarray,
        k: int,
        filter_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        """Return top-k results as list of {**meta, score: float}.

        kind: 'chapter' | 'section' | 'paragraph'
        filter_ids: if set, restrict search to these idx values.
        """
        ...
