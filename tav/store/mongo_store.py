"""MongoDB index store — supports Atlas Vector Search or FAISS fallback."""

import numpy as np
from typing import Optional

try:
    import pymongo
except ImportError:
    pymongo = None

KINDS = ("chapter", "section", "paragraph")


class MongoStore:
    """Stores TAV indices in MongoDB.

    For vector search:
      - If Atlas Vector Search index exists on the `nodes` collection, queries use $vectorSearch.
      - Otherwise, vectors are pulled into a temp FAISS index at query time.

    atlas_vector_index: name of the Atlas Search index (default: "vector_index").
                        Set to None to force FAISS fallback.
    """

    def __init__(self, uri: str, db_name: str = "tav", atlas_vector_index: str = "vector_index"):
        if pymongo is None:
            raise ImportError("Install pymongo: pip install pymongo")
        self.client = pymongo.MongoClient(uri)
        self.db = self.client[db_name]
        self.docs = self.db["documents"]
        self.nodes = self.db["nodes"]
        self.atlas_idx = atlas_vector_index
        self._ensure_indexes()

    def _ensure_indexes(self):
        self.docs.create_index("name", unique=True)
        self.nodes.create_index([("document_id", 1), ("kind", 1)])
        self.nodes.create_index([("document_id", 1), ("kind", 1), ("idx", 1)], unique=True)

    # ---- persistence ----

    def save(self, name: str, index_data: dict) -> str:
        cfg = index_data["config"]

        doc = self.docs.find_one_and_update(
            {"name": name},
            {"$set": {
                "embed_model": cfg["embed_model"],
                "dim": cfg["dim"],
                "weights": cfg["weights"],
            }},
            upsert=True,
            return_document=pymongo.ReturnDocument.AFTER,
        )
        doc_id = doc["_id"]

        # Clear old nodes
        self.nodes.delete_many({"document_id": doc_id})

        # Bulk insert
        batch = []
        for kind in KINDS:
            ix = index_data[f"{kind}_index"]
            meta_list = index_data[f"{kind}_meta"]
            for i, meta in enumerate(meta_list):
                vec = ix.reconstruct(i).tolist() if ix.ntotal > 0 else None
                batch.append({
                    "document_id": doc_id,
                    "kind": kind,
                    "node_id": meta["node_id"],
                    "idx": meta.get(f"{kind}_idx", i),
                    "chapter_idx": meta.get("chapter_idx"),
                    "section_idx": meta.get("section_idx"),
                    "title": meta["title"],
                    "page_start": meta.get("page_start"),
                    "page_end": meta.get("page_end"),
                    "level": meta.get("level"),
                    "body": meta.get("text"),
                    "embedding": vec,
                })
        if batch:
            self.nodes.insert_many(batch)

        return str(doc_id)

    def _doc_id(self, name: str):
        doc = self.docs.find_one({"name": name}, {"_id": 1})
        if not doc:
            raise KeyError(f"Document '{name}' not found")
        return doc["_id"]

    def load(self, name: str) -> dict:
        """Reconstruct the same dict shape as build_index, including FAISS indices."""
        import faiss

        doc = self.docs.find_one({"name": name})
        if not doc:
            raise KeyError(f"Document '{name}' not found")
        doc_id = doc["_id"]
        dim = doc["dim"]

        res = {
            "config": {
                "embed_model": doc["embed_model"], "dim": dim,
                "weights": doc["weights"],
            },
        }

        for kind in KINDS:
            cursor = self.nodes.find(
                {"document_id": doc_id, "kind": kind},
            ).sort("idx", 1)

            meta_list, vecs = [], []
            for r in cursor:
                m = {
                    "node_id": r["node_id"], f"{kind}_idx": r["idx"],
                    "chapter_idx": r.get("chapter_idx"), "title": r["title"],
                    "page_start": r.get("page_start"), "page_end": r.get("page_end"),
                    "level": r.get("level"),
                }
                if kind != "chapter":
                    m["section_idx"] = r.get("section_idx")
                if kind == "paragraph":
                    m["paragraph_idx"] = r["idx"]
                    m["text"] = r.get("body")
                meta_list.append(m)
                if r.get("embedding"):
                    vecs.append(np.array(r["embedding"], dtype=np.float32))

            ix = faiss.IndexFlatIP(dim)
            if vecs:
                ix.add(np.stack(vecs))

            res[f"{kind}_index"] = ix
            res[f"{kind}_meta"] = meta_list

        counts = {f"num_{k}s": len(res[f"{k}_meta"]) for k in KINDS}
        res["config"].update(counts)
        return res

    def delete(self, name: str) -> None:
        doc = self.docs.find_one({"name": name}, {"_id": 1})
        if doc:
            self.nodes.delete_many({"document_id": doc["_id"]})
            self.docs.delete_one({"_id": doc["_id"]})

    def list_documents(self) -> list[dict]:
        return [
            {"name": d["name"], "embed_model": d["embed_model"],
             "dim": d["dim"], "weights": d["weights"]}
            for d in self.docs.find().sort("name", 1)
        ]

    def search_vectors(
        self,
        name: str,
        kind: str,
        query_vec: np.ndarray,
        k: int,
        filter_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        vec = query_vec[0].tolist() if query_vec.ndim == 2 else query_vec.tolist()
        doc_id = self._doc_id(name)

        # Try Atlas vector search first
        if self.atlas_idx:
            try:
                return self._atlas_search(doc_id, kind, vec, k, filter_ids)
            except pymongo.errors.OperationFailure:
                pass  # fallback to FAISS

        return self._faiss_search(doc_id, kind, query_vec, k, filter_ids)

    def _atlas_search(self, doc_id, kind, vec, k, filter_ids):
        pre_filter = {"document_id": doc_id, "kind": kind}
        if filter_ids is not None:
            pre_filter["idx"] = {"$in": filter_ids}

        pipeline = [
            {"$vectorSearch": {
                "index": self.atlas_idx,
                "path": "embedding",
                "queryVector": vec,
                "numCandidates": max(k * 10, 100),
                "limit": k,
                "filter": pre_filter,
            }},
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        ]
        results = []
        for r in self.nodes.aggregate(pipeline):
            m = {
                "node_id": r["node_id"], f"{kind}_idx": r["idx"],
                "chapter_idx": r.get("chapter_idx"), "title": r["title"],
                "page_start": r.get("page_start"), "page_end": r.get("page_end"),
                "level": r.get("level"), "score": r["score"],
            }
            if kind != "chapter":
                m["section_idx"] = r.get("section_idx")
            if kind == "paragraph":
                m["paragraph_idx"] = r["idx"]
                m["text"] = r.get("body")
            results.append(m)
        return results

    def _faiss_search(self, doc_id, kind, query_vec, k, filter_ids):
        """Fallback: pull vectors from Mongo, build temp FAISS index."""
        import faiss

        filt = {"document_id": doc_id, "kind": kind}
        if filter_ids is not None:
            filt["idx"] = {"$in": filter_ids}

        rows = list(self.nodes.find(filt).sort("idx", 1))
        if not rows:
            return []

        meta_list, vecs = [], []
        for r in rows:
            m = {
                "node_id": r["node_id"], f"{kind}_idx": r["idx"],
                "chapter_idx": r.get("chapter_idx"), "title": r["title"],
                "page_start": r.get("page_start"), "page_end": r.get("page_end"),
                "level": r.get("level"),
            }
            if kind != "chapter":
                m["section_idx"] = r.get("section_idx")
            if kind == "paragraph":
                m["paragraph_idx"] = r["idx"]
                m["text"] = r.get("body")
            meta_list.append(m)
            vecs.append(np.array(r["embedding"], dtype=np.float32))

        mat = np.stack(vecs)
        dim = mat.shape[1]
        ix = faiss.IndexFlatIP(dim)
        ix.add(mat)
        k = min(k, ix.ntotal)
        scores, ids = ix.search(query_vec if query_vec.ndim == 2 else query_vec.reshape(1, -1), k)
        return [
            {**meta_list[i], "score": float(s)}
            for s, i in zip(scores[0], ids[0]) if i >= 0
        ]
