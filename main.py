"""
FastAPI application for searching Rust library documentation stored in a Qdrant vector DB
with Redis caching.

Environment variables:
- APP_HOST (default: 0.0.0.0)
- APP_PORT (default: 8000)
- REDIS_URL (optional, e.g. redis://localhost:6379/0). If absent, caching is disabled.
- CACHE_TTL_SECONDS (default: 300)
- QDRANT_HOST (default: localhost)
- QDRANT_PORT (default: 6333)
- QDRANT_API_KEY (optional)
- QDRANT_COLLECTION (default: rust_docs)
- EMBEDDING_DIM (default: 128) — used by the simple built‑in hash embedder.

Run locally:
  uvicorn main:app --reload

Example:
  GET /health
  GET /search?query=how%20to%20use%20tokio&library=tokio&limit=5
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

# Optional imports: make the app usable even if redis/qdrant packages are not installed
try:  # redis is optional
    import redis  # type: ignore
except Exception:  # pragma: no cover - if not installed, we degrade gracefully
    redis = None  # type: ignore

try:  # qdrant is optional
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http import models as qmodels  # type: ignore
except Exception:  # pragma: no cover
    QdrantClient = None  # type: ignore
    qmodels = None  # type: ignore

# Optional embedding model for semantic search
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


class Settings:
    def __init__(self) -> None:
        self.app_host: str = os.environ.get("APP_HOST", "0.0.0.0")
        self.app_port: int = int(os.environ.get("APP_PORT", "8000"))
        self.redis_url: Optional[str] = os.environ.get("REDIS_URL")
        self.cache_ttl_seconds: int = int(os.environ.get("CACHE_TTL_SECONDS", "300"))
        self.qdrant_host: str = os.environ.get("QDRANT_HOST", "localhost")
        self.qdrant_port: int = int(os.environ.get("QDRANT_PORT", "6333"))
        self.qdrant_api_key: Optional[str] = os.environ.get("QDRANT_API_KEY")
        self.qdrant_collection: str = os.environ.get("QDRANT_COLLECTION", "rust_docs")
        self.embedding_dim: int = int(os.environ.get("EMBEDDING_DIM", "128"))


def get_settings() -> Settings:
    return Settings()


def get_redis_client(settings: Settings = Depends(get_settings)):
    if settings.redis_url and redis is not None:
        try:
            client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
            # Do not ping here to avoid slowing startup; ping in health-check
            return client
        except Exception:
            return None
    return None


def get_qdrant_client(settings: Settings = Depends(get_settings)):
    if QdrantClient is None:
        return None
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            timeout=2.0,
        )
        return client
    except Exception:
        return None


# Semantic embedder: prefer SentenceTransformer if available, otherwise fall back to deterministic hash
_ST_MODEL = None  # cached SentenceTransformer


def _get_st_model():
    global _ST_MODEL
    if SentenceTransformer is None:
        return None
    if _ST_MODEL is None:
        try:
            # Align with rag_system.py default model
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _ST_MODEL = None
    return _ST_MODEL


def embed_text(text: str, dim: int) -> List[float]:
    """Return a semantic embedding if possible; otherwise a deterministic hash vector.

    Note: When SentenceTransformer is available, we ignore the 'dim' parameter and
    return the model's native dimension (e.g., 384 for all-MiniLM-L6-v2), which
    must match the Qdrant collection. This aligns with how the DB is filled in rag_system.py.
    """
    model = _get_st_model()
    if model is not None:
        try:
            vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
            return vec.astype(float).tolist()
        except Exception:
            # If embedding fails unexpectedly, fall back to hash for robustness
            pass

    # Fallback: Simple deterministic embedder so we don't depend on external ML libs
    # It maps the sha256 of the input into a fixed-size vector.
    if dim <= 0:
        raise ValueError("Embedding dimension must be positive")
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Repeat hash to cover required dimension
    bytes_needed = dim
    buf = (h * ((bytes_needed // len(h)) + 1))[:bytes_needed]
    # Scale bytes [0..255] to floats [0..1]
    return [b / 255.0 for b in buf]


app = FastAPI(title="Rust Docs Search API", version="0.1.0")


@app.get("/health")
def health(
    settings: Settings = Depends(get_settings),
    r=Depends(get_redis_client),
    q=Depends(get_qdrant_client),
):
    status: Dict[str, Any] = {"status": "ok"}

    # Redis health
    redis_info: Dict[str, Any] = {"configured": bool(settings.redis_url), "available": False}
    if r is not None:
        try:
            pong = r.ping()
            redis_info["available"] = bool(pong)
        except Exception as e:  # pragma: no cover - depends on external service
            redis_info["error"] = str(e)
    status["redis"] = redis_info

    # Qdrant health
    qdrant_info: Dict[str, Any] = {"configured": True, "available": False, "collection": settings.qdrant_collection}
    if q is not None and qmodels is not None:
        try:
            colls = q.get_collections()
            names = [c.name for c in getattr(colls, "collections", [])]
            qdrant_info["available"] = True
            qdrant_info["collections"] = names
            qdrant_info["collection_exists"] = settings.qdrant_collection in set(names)
        except Exception as e:  # pragma: no cover
            qdrant_info["error"] = str(e)
    else:
        if QdrantClient is None:
            qdrant_info["configured"] = False
    status["qdrant"] = qdrant_info

    return JSONResponse(status_code=200, content=status)


@app.get("/search")
def search(
    query: str = Query(..., min_length=1, description="User search query"),
    library: Optional[str] = Query(None, description="Optional library name to filter by payload field 'library'"),
    limit: int = Query(5, ge=1, le=50),
    settings: Settings = Depends(get_settings),
    r=Depends(get_redis_client),
    q=Depends(get_qdrant_client),
):
    if q is None or qmodels is None:
        raise HTTPException(status_code=503, detail="Vector search backend (Qdrant) is not available")

    # Try cache first
    cache_key = _make_cache_key(settings.qdrant_collection, query, library, limit)
    if r is not None:
        try:
            cached = r.get(cache_key)
        except Exception:
            cached = None
        if cached:
            try:
                data = json.loads(cached)
                data["cached"] = True
                return JSONResponse(status_code=200, content=data)
            except Exception:
                # Corrupted cache – ignore
                pass

    # Build query vector
    vector = embed_text(query, settings.embedding_dim)

    # Optional payload filter by library
    payload_filter = None
    if library:
        payload_filter = qmodels.Filter(must=[qmodels.FieldCondition(key="library", match=qmodels.MatchValue(value=library))])

    try:
        results = q.search(
            collection_name=settings.qdrant_collection,
            query_vector=vector,
            limit=limit,
            query_filter=payload_filter if payload_filter is not None else None,
            with_payload=True,
            with_vectors=False,
        )
    except TypeError:
        # Older qdrant-client uses 'filter' instead of 'query_filter'
        results = q.search(
            collection_name=settings.qdrant_collection,
            query_vector=vector,
            limit=limit,
            filter=payload_filter if payload_filter is not None else None,
            with_payload=True,
            with_vectors=False,
        )

    items = [
        {
            "id": getattr(p, "id", None),
            "score": getattr(p, "score", None),
            "payload": getattr(p, "payload", None),
        }
        for p in (results or [])
    ]

    response = {"query": query, "library": library, "limit": limit, "results": items, "cached": False}

    # Store in cache
    if r is not None and items:
        try:
            r.setex(cache_key, settings.cache_ttl_seconds, json.dumps(response))
        except Exception:
            pass

    return JSONResponse(status_code=200, content=response)


@app.post("/cache/flush")
def flush_cache(
    settings: Settings = Depends(get_settings),
    r=Depends(get_redis_client),
):
    if r is None:
        raise HTTPException(status_code=503, detail="Redis cache is not configured or unavailable")
    try:
        # Instead of FLUSHDB, delete keys by pattern
        pattern = f"search:{settings.qdrant_collection}:*"
        cursor = 0
        deleted = 0
        
        try:
            before = int(r.dbsize())
        except Exception:
            before = None
        
        # Use SCAN to find and delete keys matching our pattern
        while True:
            cursor, keys = r.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                deleted += r.delete(*keys)
            if cursor == 0:
                break
        
        try:
            after = int(r.dbsize())
        except Exception:
            after = None
            
        return JSONResponse(status_code=200, content={
            "status": "ok",
            "flushed": True,
            "deleted_keys": deleted,
            "before": before,
            "after": after,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to flush Redis cache: {e}")


def _make_cache_key(collection: str, query: str, library: Optional[str], limit: int) -> str:
    qh = hashlib.sha1(query.encode("utf-8")).hexdigest()
    lib = library or "*"
    return f"search:{collection}:{lib}:{limit}:{qh}"


# Optional: allow running with `python main.py`
if __name__ == "__main__":
    import uvicorn  # type: ignore

    settings = get_settings()
    uvicorn.run("main:app", host=settings.app_host, port=settings.app_port, reload=False)
