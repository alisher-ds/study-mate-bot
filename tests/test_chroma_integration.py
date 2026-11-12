"""
rag_engine against the REAL ChromaDB (PersistentClient in a temporary directory).

No network and no embedding model: vectors are made by hand from keywords, so "nearest chunk"
is predictable. Chroma's anonymous telemetry is switched off for the same reason.
"""
import asyncio

import chromadb
import pytest
from chromadb.config import Settings

import rag_engine

USER = 7
OTHER_USER = 8

# One axis per topic; a text gets a 1.0 on the axis of every topic word it contains.
TOPICS = ["mushuk", "itlar", "avtomobil"]


class HandMadeVectors:
    """Stand-in for the embedding model: a vector is derived from the topic words in the text."""

    max_seq_length = 128

    def encode(self, texts, **kwargs):
        return [
            [1.0 if topic in text.lower() else 0.0 for topic in TOPICS] + [0.01]
            for text in texts
        ]


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def client(tmp_path):
    client = chromadb.PersistentClient(
        path=str(tmp_path / "chroma_db"), settings=Settings(anonymized_telemetry=False)
    )
    yield client
    # Release Chroma's hold on the sqlite file so Windows can delete the temp directory.
    client._system.stop()
    chromadb.api.client.SharedSystemClient.clear_system_cache()


@pytest.fixture
def rag(monkeypatch, client):
    monkeypatch.setattr(rag_engine, "get_chroma_client", lambda: client)
    monkeypatch.setattr(rag_engine, "get_embedding_model", lambda: HandMadeVectors())
    return rag_engine


CHUNKS = [
    "Mushuklar sutemizuvchi hayvonlardir.",
    "Itlar insonning sodiq do'sti.",
    "Avtomobil to'rt g'ildirakli transport vositasi.",
]


def upload(rag, chunks=CHUNKS, doc="hayvonlar.pdf", user_id=USER):
    return run(rag.add_document(user_id, chunks, doc))


# ---------------------------------------------------------------- add / query / get

def test_search_returns_the_nearest_chunk_with_its_source(rag):
    assert upload(rag) == 3

    found = run(rag.search_relevant_chunks(USER, "mushuk haqida", top_k=1))

    assert found == [{"text": CHUNKS[0], "source": "hayvonlar.pdf"}]


def test_search_asking_for_more_than_stored_returns_everything(rag):
    upload(rag)
    assert len(run(rag.search_relevant_chunks(USER, "itlar", top_k=10))) == 3


def test_search_without_any_upload_is_empty_and_does_not_fail(rag):
    assert run(rag.search_relevant_chunks(USER, "mushuk")) == []


def test_sample_returns_chunks_in_document_order_for_quiz_and_summary(rag):
    upload(rag, [f"mushuk {i:03d}" for i in range(20)], doc="a.pdf")
    upload(rag, [f"itlar {i:03d}" for i in range(20)], doc="b.pdf")

    sample = rag._sample_sync(USER, 5)

    assert len(sample) == 5
    assert [item["source"] for item in sample] == sorted(item["source"] for item in sample)
    assert sample[0]["text"] == "mushuk 000"  # first chunk of the first document
    assert sample[-1]["text"] == "itlar 019"  # last chunk of the last document


def test_sample_without_any_upload_is_empty(rag):
    assert rag._sample_sync(USER, 5) == []


def test_many_chunks_are_stored_in_batches(rag, client):
    total = rag.CHROMA_ADD_BATCH * 2 + 100
    upload(rag, [f"mushuk bo'lak {i}" for i in range(total)])

    assert rag.get_or_create_collection(USER).count() == total


# ---------------------------------------------------------------- replace / delete

def test_uploading_the_same_file_name_again_replaces_it(rag):
    upload(rag, ["mushuk eski matn bir", "mushuk eski matn ikki"])
    upload(rag, ["itlar yangi matn"])

    stored = rag.get_or_create_collection(USER).get(include=["documents"])["documents"]

    assert stored == ["itlar yangi matn"]


def test_different_file_names_live_side_by_side(rag):
    upload(rag, ["mushuk matn"], doc="a.pdf")
    upload(rag, ["itlar matn"], doc="b.pdf")

    collection = rag.get_or_create_collection(USER)

    assert collection.count() == 2
    assert collection.get(where={"source": "a.pdf"}, include=[])["ids"]
    assert collection.get(where={"source": "zzz.pdf"}, include=[])["ids"] == []


def test_deleting_chunks_by_id(rag):
    upload(rag)
    collection = rag.get_or_create_collection(USER)
    ids = collection.get(where={"source": "hayvonlar.pdf"}, include=[])["ids"]

    collection.delete(ids=ids[:1])

    assert collection.count() == 2


# ---------------------------------------------------------------- /reset

def test_reset_deletes_the_users_collection(rag, client):
    upload(rag)

    assert run(rag.reset_user_data(USER)) is True

    assert f"user_{USER}" not in [c.name for c in client.list_collections()]
    assert run(rag.search_relevant_chunks(USER, "mushuk")) == []


def test_reset_with_nothing_stored_returns_false_without_an_error(rag):
    assert run(rag.reset_user_data(USER)) is False


def test_reset_twice_returns_false_the_second_time(rag):
    upload(rag)
    assert run(rag.reset_user_data(USER)) is True
    assert run(rag.reset_user_data(USER)) is False


def test_reset_works_on_an_emptied_collection(rag):
    # Collection exists but holds no chunks (e.g. after a search created it).
    run(rag.search_relevant_chunks(USER, "mushuk"))
    assert run(rag.reset_user_data(USER)) is True


def test_upload_works_again_after_reset(rag):
    upload(rag)
    run(rag.reset_user_data(USER))

    assert upload(rag) == 3
    assert len(run(rag.search_relevant_chunks(USER, "avtomobil", top_k=1))) == 1


def test_reset_leaves_other_users_alone(rag):
    upload(rag, user_id=USER)
    upload(rag, user_id=OTHER_USER)

    run(rag.reset_user_data(USER))

    assert len(run(rag.search_relevant_chunks(OTHER_USER, "itlar", top_k=10))) == 3
