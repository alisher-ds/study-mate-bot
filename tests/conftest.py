"""Shared fakes: no network, no model download, no real ChromaDB."""
import httpx
import pytest

import rag_engine


class StubTokenizer:
    """Offline stand-in for a HF tokenizer: every word costs one token per 3 characters."""

    def __call__(self, text, add_special_tokens=False, verbose=True):
        ids = [w[i:i + 3] for w in text.split() for i in range(0, len(w), 3)]
        return {"input_ids": ids}


class FakeEmbeddingModel:
    def __init__(self):
        self.calls = []
        self.max_seq_length = 128
        self.tokenizer = StubTokenizer()

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]


class FakeCollection:
    """Minimal in-memory stand-in for a Chroma collection (insertion-ordered)."""

    def __init__(self):
        self.rows = {}  # id -> (embedding, metadata, document)
        self.add_calls = 0

    def count(self):
        return len(self.rows)

    def add(self, ids, embeddings, metadatas, documents):
        duplicates = [i for i in ids if i in self.rows]
        if duplicates:
            raise ValueError(f"duplicate ids: {duplicates}")
        self.add_calls += 1
        for row in zip(ids, embeddings, metadatas, documents, strict=False):
            self.rows[row[0]] = row[1:]

    def get(self, where=None, include=None):
        selected = [
            (i, row) for i, row in self.rows.items()
            if where is None or all(row[1].get(k) == v for k, v in where.items())
        ]
        result = {"ids": [i for i, _ in selected]}
        include = include or []
        if "documents" in include:
            result["documents"] = [row[2] for _, row in selected]
        if "metadatas" in include:
            result["metadatas"] = [row[1] for _, row in selected]
        return result

    def query(self, query_embeddings, n_results):
        rows = list(self.rows.values())[:n_results]
        return {
            "documents": [[row[2] for row in rows]],
            "metadatas": [[row[1] for row in rows]],
        }

    def delete(self, ids):
        for i in ids:
            del self.rows[i]


class FakeChromaClient:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name):
        return self.collections.setdefault(name, FakeCollection())

    def delete_collection(self, name):
        if name not in self.collections:
            raise ValueError(f"Collection {name} does not exist.")
        del self.collections[name]


class FakeGroq:
    """chat.completions.create() replays `script`: exceptions are raised, strings returned."""

    def __init__(self):
        self.script = ["fake answer"]
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, Exception):
            raise step
        message = type("Message", (), {"content": step})()
        return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()


def http_response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://api.groq.test/chat"))


@pytest.fixture
def embedder():
    return FakeEmbeddingModel()


@pytest.fixture(autouse=True)
def no_real_embedding_model(monkeypatch):
    """Safety net: no test may download or load the real sentence-transformers model."""
    monkeypatch.setattr(rag_engine, "get_embedding_model", lambda: FakeEmbeddingModel())


@pytest.fixture
def chroma():
    return FakeChromaClient()


@pytest.fixture
def groq_client():
    return FakeGroq()


@pytest.fixture
def rag(monkeypatch, embedder, chroma, groq_client):
    """rag_engine wired to the fakes above, with retry sleeps disabled."""
    monkeypatch.setattr(rag_engine, "get_embedding_model", lambda: embedder)
    monkeypatch.setattr(rag_engine, "get_chroma_client", lambda: chroma)
    monkeypatch.setattr(rag_engine, "get_groq_client", lambda: groq_client)
    monkeypatch.setattr(rag_engine, "LLM_RETRY_DELAY", 0)
    return rag_engine
