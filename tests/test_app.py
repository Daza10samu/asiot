import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

import main


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        # emulate redis decode_responses=True behavior: values are strings
        self.store[key] = value

    def ping(self):
        return True


class FakePoint:
    def __init__(self, pid, score, payload):
        self.id = pid
        self.score = score
        self.payload = payload


class FakeQdrantClient:
    def search(self, **kwargs):
        limit = kwargs.get("limit", 5)
        return [
            FakePoint(pid=i, score=1.0 / (i + 1), payload={"library": "tokio", "text": f"doc {i}"})
            for i in range(limit)
        ]

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name="rust_docs")])


# Minimal qmodels stub to satisfy the search endpoint
class _Filter:
    def __init__(self, must=None):
        self.must = must or []


class _FieldCondition:
    def __init__(self, key, match):
        self.key = key
        self.match = match


class _MatchValue:
    def __init__(self, value):
        self.value = value


qmodels_stub = SimpleNamespace(Filter=_Filter, FieldCondition=_FieldCondition, MatchValue=_MatchValue)


def test_health_with_no_backends():
    # Ensure qmodels exists to avoid configured flag confusion
    main.qmodels = qmodels_stub

    client = TestClient(main.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    # Redis and Qdrant keys are present
    assert "redis" in data and "qdrant" in data


def test_search_uses_cache(monkeypatch):
    # Override dependencies
    main.qmodels = qmodels_stub

    fake_redis = FakeRedis()
    fake_qdrant = FakeQdrantClient()

    app = main.app

    # Dependency overrides
    app.dependency_overrides[main.get_redis_client] = lambda: fake_redis
    app.dependency_overrides[main.get_qdrant_client] = lambda: fake_qdrant

    client = TestClient(app)

    # First call fills cache
    r1 = client.get("/search", params={"query": "hello", "limit": 3})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["cached"] is False
    assert len(d1["results"]) == 3

    # Second call should be served from cache
    r2 = client.get("/search", params={"query": "hello", "limit": 3})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["cached"] is True
    assert d2["results"] == d1["results"]

    # Clean up overrides
    app.dependency_overrides.clear()
