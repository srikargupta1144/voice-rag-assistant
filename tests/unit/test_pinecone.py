"""Pinecone backend against a mocked client (no network, no API key)."""

from unittest.mock import MagicMock

import numpy as np

from retrieval.vectorstore import PineconeStore


def _store(has_index=True):
    client = MagicMock()
    client.has_index.return_value = has_index
    store = PineconeStore(dim=4, index_name="idx", namespace="ns", client=client)
    return store, client, client.Index.return_value


def test_creates_index_when_missing():
    _, client, _ = _store(has_index=False)
    kwargs = client.create_index.call_args.kwargs
    assert kwargs["name"] == "idx" and kwargs["dimension"] == 4 and kwargs["metric"] == "cosine"


def test_existing_index_not_recreated():
    _, client, _ = _store(has_index=True)
    client.create_index.assert_not_called()


def test_upsert_batches_with_metadata(chunks):
    store, _, index = _store()
    store.UPSERT_BATCH = 3
    store.add(chunks, np.ones((len(chunks), 4), dtype=np.float32))
    assert index.upsert.call_count == 2
    first = index.upsert.call_args_list[0].kwargs
    assert first["namespace"] == "ns"
    rec = first["vectors"][0]
    assert rec["id"] == chunks[0].chunk_id
    assert rec["metadata"]["page"] == -1  # None is not allowed in Pinecone metadata
    assert rec["metadata"]["text"] == chunks[0].text


def test_search_maps_matches_back_to_chunks():
    store, _, index = _store()
    index.query.return_value = {
        "matches": [
            {
                "id": "p.pdf:4:0",
                "score": 0.91,
                "metadata": {
                    "text": "Flood is excluded.",
                    "source": "p.pdf",
                    "page": 4,
                    "section": "Exclusions",
                },
            },
            {
                "id": "f.md:0:1",
                "score": 0.5,
                "metadata": {"text": "Wires", "source": "f.md", "page": -1, "section": "Wires"},
            },
        ]
    }
    hits = store.search(np.ones(4, dtype=np.float32), 2)
    assert index.query.call_args.kwargs["top_k"] == 2
    assert index.query.call_args.kwargs["namespace"] == "ns"
    assert hits[0].chunk.page == 4 and hits[0].scores["dense"] == 0.91
    assert hits[1].chunk.page is None
    assert hits[0].chunk.label() == "p.pdf, p. 4"


def test_count_reads_namespace_stats():
    store, _, index = _store()
    index.describe_index_stats.return_value = {"namespaces": {"ns": {"vector_count": 7}}}
    assert store.count() == 7
