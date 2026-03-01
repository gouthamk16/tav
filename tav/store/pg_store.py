"""PostgreSQL + pgvector index store."""

import json
import numpy as np
from typing import Optional

try:
    import psycopg
    from pgvector.psycopg import register_vector
except ImportError:
    psycopg = None


KINDS = ("chapter", "section", "paragraph")

_INIT_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tav_documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL UNIQUE,
    embed_model TEXT NOT NULL,
    dim         INT  NOT NULL,
    weights     REAL[] NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tav_nodes (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id   UUID NOT NULL REFERENCES tav_documents(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL CHECK (kind IN ('chapter','section','paragraph')),
    node_id       TEXT NOT NULL,
    idx           INT  NOT NULL,
    chapter_idx   INT,
    section_idx   INT,
    title         TEXT NOT NULL,
    page_start    INT,
    page_end      INT,
    level         INT,
    body          TEXT,
    embedding     vector
);

CREATE INDEX IF NOT EXISTS ix_tav_nodes_doc_kind ON tav_nodes (document_id, kind);
CREATE UNIQUE INDEX IF NOT EXISTS ix_tav_nodes_unique ON tav_nodes (document_id, kind, idx);
"""


class PgStore:
    """Stores TAV indices in PostgreSQL with pgvector for in-DB similarity search."""

    def __init__(self, conninfo: str):
        if psycopg is None:
            raise ImportError("Install psycopg and pgvector: pip install 'psycopg[binary]' pgvector")
        self.conninfo = conninfo
        self._ensure_schema()

    def _conn(self):
        conn = psycopg.connect(self.conninfo, autocommit=True)
        register_vector(conn)
        return conn

    def _ensure_schema(self):
        with self._conn() as conn:
            conn.execute(_INIT_SQL)

    # ---- persistence ----

    def save(self, name: str, index_data: dict) -> str:
        cfg = index_data["config"]
        dim = cfg["dim"]

        with self._conn() as conn:
            # Upsert document
            row = conn.execute(
                """INSERT INTO tav_documents (name, embed_model, dim, weights)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (name) DO UPDATE
                     SET embed_model = EXCLUDED.embed_model,
                         dim = EXCLUDED.dim,
                         weights = EXCLUDED.weights
                   RETURNING id""",
                (name, cfg["embed_model"], dim, cfg["weights"]),
            ).fetchone()
            doc_id = row[0]

            # Clear old nodes for this doc
            conn.execute("DELETE FROM tav_nodes WHERE document_id = %s", (doc_id,))

            # Insert all nodes
            for kind in KINDS:
                ix = index_data[f"{kind}_index"]
                meta_list = index_data[f"{kind}_meta"]
                for i, meta in enumerate(meta_list):
                    vec = ix.reconstruct(i).tolist() if ix.ntotal > 0 else None
                    conn.execute(
                        """INSERT INTO tav_nodes
                           (document_id, kind, node_id, idx, chapter_idx, section_idx,
                            title, page_start, page_end, level, body, embedding)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (
                            doc_id, kind, meta["node_id"], meta.get(f"{kind}_idx", i),
                            meta.get("chapter_idx"), meta.get("section_idx"),
                            meta["title"], meta.get("page_start"), meta.get("page_end"),
                            meta.get("level"), meta.get("text"),
                            np.array(vec, dtype=np.float32) if vec else None,
                        ),
                    )
        return str(doc_id)

    def _doc_id(self, conn, name: str):
        row = conn.execute("SELECT id FROM tav_documents WHERE name = %s", (name,)).fetchone()
        if not row:
            raise KeyError(f"Document '{name}' not found")
        return row[0]

    def load(self, name: str) -> dict:
        """Load into the same dict shape as build_index — reconstructs FAISS indices from stored vectors."""
        import faiss

        with self._conn() as conn:
            doc_id = self._doc_id(conn, name)
            doc = conn.execute(
                "SELECT embed_model, dim, weights FROM tav_documents WHERE id = %s", (doc_id,)
            ).fetchone()
            embed_model, dim, weights = doc

            res = {
                "config": {
                    "embed_model": embed_model, "dim": dim, "weights": list(weights),
                },
            }

            for kind in KINDS:
                rows = conn.execute(
                    """SELECT node_id, idx, chapter_idx, section_idx,
                              title, page_start, page_end, level, body, embedding
                       FROM tav_nodes
                       WHERE document_id = %s AND kind = %s
                       ORDER BY idx""",
                    (doc_id, kind),
                ).fetchall()

                meta_list, vecs = [], []
                for r in rows:
                    m = {
                        "node_id": r[0], f"{kind}_idx": r[1],
                        "chapter_idx": r[2], "title": r[4],
                        "page_start": r[5], "page_end": r[6], "level": r[7],
                    }
                    if kind != "chapter":
                        m["section_idx"] = r[3]
                    if kind == "paragraph":
                        m["paragraph_idx"] = r[1]
                        m["text"] = r[8]
                    meta_list.append(m)
                    if r[9] is not None:
                        vecs.append(np.array(r[9], dtype=np.float32))

                ix = faiss.IndexFlatIP(dim)
                if vecs:
                    ix.add(np.stack(vecs))

                res[f"{kind}_index"] = ix
                res[f"{kind}_meta"] = meta_list

            counts = {f"num_{k}s": len(res[f"{k}_meta"]) for k in KINDS}
            res["config"].update(counts)

        return res

    def delete(self, name: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM tav_documents WHERE name = %s", (name,))

    def list_documents(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT name, embed_model, dim, weights, created_at FROM tav_documents ORDER BY created_at"
            ).fetchall()
        return [
            {"name": r[0], "embed_model": r[1], "dim": r[2], "weights": list(r[3]), "created_at": str(r[4])}
            for r in rows
        ]

    def search_vectors(
        self,
        name: str,
        kind: str,
        query_vec: np.ndarray,
        k: int,
        filter_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        """In-DB similarity search via pgvector inner product."""
        vec = query_vec[0] if query_vec.ndim == 2 else query_vec
        vec_param = np.array(vec, dtype=np.float32)

        with self._conn() as conn:
            doc_id = self._doc_id(conn, name)

            conditions = ["document_id = %s", "kind = %s"]
            where_params: list = [doc_id, kind]

            if filter_ids is not None:
                conditions.append("idx = ANY(%s)")
                where_params.append(filter_ids)

            where = " AND ".join(conditions)

            # Param order must match placeholder order: SELECT vec, WHERE params, ORDER vec, LIMIT k
            rows = conn.execute(
                f"""SELECT node_id, idx, chapter_idx, section_idx,
                           title, page_start, page_end, level, body,
                           (embedding <#> %s) * -1 AS score
                    FROM tav_nodes
                    WHERE {where}
                    ORDER BY embedding <#> %s
                    LIMIT %s""",
                (vec_param, *where_params, vec_param, k),
            ).fetchall()

            results = []
            for r in rows:
                m = {
                    "node_id": r[0], f"{kind}_idx": r[1],
                    "chapter_idx": r[2], "title": r[4],
                    "page_start": r[5], "page_end": r[6], "level": r[7],
                    "score": float(r[9]),
                }
                if kind != "chapter":
                    m["section_idx"] = r[3]
                if kind == "paragraph":
                    m["paragraph_idx"] = r[1]
                    m["text"] = r[8]
                results.append(m)
            return results
