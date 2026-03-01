"""File-based index store — wraps the existing FAISS + JSON on-disk layout."""

import os
import json
import numpy as np
import faiss
from typing import Optional


KINDS = ("chapter", "section", "paragraph")


class FileStore:
    """Stores indices as .faiss + _meta.json files in a local directory."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir

    def _dir(self, name: str) -> str:
        return os.path.join(self.base_dir, f".tav_index_{name}")

    def save(self, name: str, index_data: dict) -> str:
        path = self._dir(name)
        os.makedirs(path, exist_ok=True)
        for kind in KINDS:
            faiss.write_index(index_data[f"{kind}_index"], os.path.join(path, f"{kind}.faiss"))
            with open(os.path.join(path, f"{kind}_meta.json"), "w", encoding="utf-8") as f:
                json.dump(index_data[f"{kind}_meta"], f, indent=2, ensure_ascii=False)
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(index_data["config"], f, indent=2, ensure_ascii=False)
        return path

    def load(self, name: str) -> dict:
        path = self._dir(name)
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        res = {"config": cfg}
        for kind in KINDS:
            res[f"{kind}_index"] = faiss.read_index(os.path.join(path, f"{kind}.faiss"))
            with open(os.path.join(path, f"{kind}_meta.json"), encoding="utf-8") as f:
                res[f"{kind}_meta"] = json.load(f)
        return res

    def delete(self, name: str) -> None:
        import shutil
        path = self._dir(name)
        if os.path.isdir(path):
            shutil.rmtree(path)

    def list_documents(self) -> list[dict]:
        prefix = ".tav_index_"
        docs = []
        if not os.path.isdir(self.base_dir):
            return docs
        for entry in os.listdir(self.base_dir):
            if entry.startswith(prefix) and os.path.isdir(os.path.join(self.base_dir, entry)):
                doc_name = entry[len(prefix):]
                cfg_path = os.path.join(self.base_dir, entry, "config.json")
                cfg = {}
                if os.path.isfile(cfg_path):
                    with open(cfg_path) as f:
                        cfg = json.load(f)
                docs.append({"name": doc_name, **cfg})
        return docs

    def search_vectors(
        self,
        name: str,
        kind: str,
        query_vec: np.ndarray,
        k: int,
        filter_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        data = self.load(name)
        ix = data[f"{kind}_index"]
        meta = data[f"{kind}_meta"]

        if filter_ids is not None:
            dim = ix.d
            vecs = np.zeros((len(filter_ids), dim), dtype=np.float32)
            for i, idx in enumerate(filter_ids):
                vecs[i] = ix.reconstruct(idx)
            tmp = faiss.IndexFlatIP(dim)
            tmp.add(vecs)
            k = min(k, len(filter_ids))
            scores, local_ids = tmp.search(query_vec, k)
            return [
                {**meta[filter_ids[lid]], "score": float(s)}
                for s, lid in zip(scores[0], local_ids[0]) if lid >= 0
            ]

        k = min(k, ix.ntotal)
        if k == 0:
            return []
        scores, ids = ix.search(query_vec, k)
        return [
            {**meta[i], "score": float(s)}
            for s, i in zip(scores[0], ids[0]) if i >= 0
        ]
