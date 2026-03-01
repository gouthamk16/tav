"""Store factory — get_store() returns the right backend."""

from .base import IndexStore
from .file_store import FileStore


def get_store(backend: str = "file", **kwargs) -> IndexStore:
    """Create a store instance.

    backend: 'file' | 'postgres' | 'mongo'
    kwargs:
        file:     base_dir (str)
        postgres: conninfo (str, e.g. 'postgresql://user:pass@host/db')
        mongo:    uri (str), db_name (str, default 'tav'),
                  atlas_vector_index (str|None, default 'vector_index')
    """
    if backend == "file":
        return FileStore(base_dir=kwargs.get("base_dir", "."))

    if backend == "postgres":
        from .pg_store import PgStore
        conninfo = kwargs.get("conninfo") or kwargs.get("uri")
        if not conninfo:
            raise ValueError("postgres backend requires 'conninfo' or 'uri'")
        return PgStore(conninfo=conninfo)

    if backend == "mongo":
        from .mongo_store import MongoStore
        uri = kwargs.get("uri")
        if not uri:
            raise ValueError("mongo backend requires 'uri'")
        return MongoStore(
            uri=uri,
            db_name=kwargs.get("db_name", "tav"),
            atlas_vector_index=kwargs.get("atlas_vector_index", "vector_index"),
        )

    raise ValueError(f"Unknown store backend: {backend}")
