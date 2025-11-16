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
- INFERENCE_URL (default: http://inference:8080) — URL of the inference service.

Run locally:
  uvicorn main:app --reload

Example:
  GET /health
  GET /search?query=how%20to%20use%20tokio&library=tokio&limit=5
  POST /predict with JSON body
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import requests
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
        # Inference service settings
        self.inference_url: str = os.environ.get("INFERENCE_URL", "http://inference:8080")


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

# Wine feature definitions from sklearn wine dataset
WINE_FEATURES = {
    "alcohol": {"index": 0, "description": "Alcohol content (%)", "range": "11.0 - 14.8"},
    "malic_acid": {"index": 1, "description": "Malic acid (g/L)", "range": "0.7 - 5.8"},
    "ash": {"index": 2, "description": "Ash content (g/L)", "range": "1.4 - 3.2"},
    "alcalinity_of_ash": {"index": 3, "description": "Alkalinity of ash", "range": "10.6 - 30.0"},
    "magnesium": {"index": 4, "description": "Magnesium (mg/L)", "range": "70 - 162"},
    "total_phenols": {"index": 5, "description": "Total phenols", "range": "0.98 - 3.88"},
    "flavanoids": {"index": 6, "description": "Flavanoids", "range": "0.34 - 5.08"},
    "nonflavanoid_phenols": {"index": 7, "description": "Non-flavanoid phenols", "range": "0.13 - 0.66"},
    "proanthocyanins": {"index": 8, "description": "Proanthocyanins", "range": "0.41 - 3.58"},
    "color_intensity": {"index": 9, "description": "Color intensity", "range": "1.3 - 13.0"},
    "hue": {"index": 10, "description": "Hue", "range": "0.48 - 1.71"},
    "od280_od315_of_diluted_wines": {"index": 11, "description": "OD280/OD315 of diluted wines", "range": "1.27 - 4.0"},
    "proline": {"index": 12, "description": "Proline content (mg/L)", "range": "278 - 1680"},
}

# Wine cultivar descriptions based on sklearn wine dataset characteristics
WINE_CULTIVARS = {
    0: {
        "name": "Cultivar 0",
        "description": "High alcohol content with rich flavanoids and color intensity",
        "characteristics": [
            "Higher alcohol content (typically 13.0-14.8%)",
            "Rich in flavanoids (antioxidants)",
            "Strong color intensity",
            "High total phenols",
            "Premium quality indicators"
        ]
    },
    1: {
        "name": "Cultivar 1", 
        "description": "Balanced wine with moderate characteristics",
        "characteristics": [
            "Moderate alcohol content (12.0-13.5%)",
            "Balanced phenolic compounds",
            "Medium color intensity",
            "Good overall balance",
            "Versatile characteristics"
        ]
    },
    2: {
        "name": "Cultivar 2",
        "description": "High proline content with unique chemical profile",
        "characteristics": [
            "Very high proline content (amino acid)",
            "Lower alcohol content (11.0-13.0%)",
            "Distinctive chemical signature",
            "Lighter color intensity",
            "Unique flavor profile"
        ]
    }
}


@app.get("/predict/info")
def predict_info():
    """
    Get information about the wine prediction model, including required features and their descriptions.
    """
    return JSONResponse(status_code=200, content={
        "model": "wine-classifier-rf",
        "description": "Random Forest classifier for wine cultivar prediction based on chemical analysis",
        "classes": WINE_CULTIVARS,
        "features": WINE_FEATURES,
        "usage": {
            "endpoint": "/predict",
            "method": "POST",
            "example": {
                "alcohol": 13.2,
                "malic_acid": 2.77,
                "ash": 2.51,
                "alcalinity_of_ash": 18.5,
                "magnesium": 96.0,
                "total_phenols": 1.9,
                "flavanoids": 0.58,
                "nonflavanoid_phenols": 0.63,
                "proanthocyanins": 1.14,
                "color_intensity": 7.5,
                "hue": 0.72,
                "od280_od315_of_diluted_wines": 1.88,
                "proline": 472.0
            }
        }
    })


@app.post("/predict")
def predict(
    data: Dict[str, Any],
    settings: Settings = Depends(get_settings),
):
    """
    Predict wine cultivar from chemical analysis data.

    Accepts either:
    1. Named features (recommended):
       {
         "alcohol": 13.2,
         "malic_acid": 2.77,
         "ash": 2.51,
         ...
       }

    2. Legacy format with feature array:
       {"instances": [[13.2, 2.77, 2.51, ...]]}

    Returns human-readable prediction with cultivar information and confidence.
    """
    inference_url = f"{settings.inference_url.rstrip('/')}/invocations"

    # Check if data is in named feature format (recommended)
    if "instances" not in data and "dataframe_split" not in data:
        # Convert named features to feature vector
        try:
            feature_vector = [None] * 13
            provided_features = []

            for feature_name, value in data.items():
                if feature_name in WINE_FEATURES:
                    idx = WINE_FEATURES[feature_name]["index"]
                    feature_vector[idx] = float(value)
                    provided_features.append(feature_name)

            # Check if all features are provided
            if None in feature_vector:
                missing = [name for name, info in WINE_FEATURES.items() 
                          if info["index"] in [i for i, v in enumerate(feature_vector) if v is None]]
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Missing required features",
                        "missing_features": missing,
                        "provided_features": provided_features,
                        "hint": "Use GET /predict/info to see all required features"
                    }
                )

            # Convert to MLflow format
            data = {"instances": [feature_vector]}

        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feature value: {str(e)}. All features must be numeric."
            )

    try:
        # Forward the request to the inference service
        response = requests.post(
            inference_url,
            json=data,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            raw_result = response.json()
            predictions = raw_result.get("predictions", [])

            # Enhance predictions with human-readable information
            enhanced_results = []
            for i, pred in enumerate(predictions):
                pred_class = int(pred)

                if pred_class not in WINE_CULTIVARS:
                    pred_class = 1  # Default to middle class if out of range

                cultivar_info = WINE_CULTIVARS[pred_class]

                enhanced_results.append({
                    "prediction": {
                        "class": pred_class,
                        "cultivar": cultivar_info["name"]
                    },
                    "description": cultivar_info["description"],
                    "characteristics": cultivar_info["characteristics"],
                    "confidence": {
                        "level": "high" if 0 <= pred_class <= 2 else "low",
                        "note": "Based on Random Forest model trained on sklearn wine dataset"
                    }
                })

            return JSONResponse(status_code=200, content={
                "success": True,
                "model": "wine-classifier-rf",
                "results": enhanced_results,
                "metadata": {
                    "samples_analyzed": len(enhanced_results),
                    "model_version": "1.0",
                    "dataset": "sklearn wine dataset (178 samples, 3 cultivars)"
                }
            })
        else:
            # Return the error from inference service
            try:
                error_detail = response.json()
            except Exception:
                error_detail = {"error": response.text}

            raise HTTPException(
                status_code=response.status_code,
                detail=f"Inference service error: {error_detail}"
            )

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to inference service at {inference_url}"
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Inference service request timed out"
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calling inference service: {str(e)}"
        )



# Optional: allow running with `python main.py`
if __name__ == "__main__":
    import uvicorn  # type: ignore

    settings = get_settings()
    uvicorn.run("main:app", host=settings.app_host, port=settings.app_port, reload=False)
