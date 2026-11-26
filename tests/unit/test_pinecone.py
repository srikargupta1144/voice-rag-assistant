"""
File: tests/unit/test_pinecone.py

Description:
    Unit tests for the Pinecone backend against a mocked client (no network, no API key).

Libraries used:
    unittest.mock - standard library; MagicMock stands in for the Pinecone client and index
    numpy - third party; test vectors
    retrieval.vectorstore - project; the PineconeStore under test

Created by: Srikar Gupta
Created date: 2026-11-18
Updated by: Srikar Gupta
Updated date: 2026-11-26

Change log:
    2026-11-18  Srikar Gupta  First version.
    2026-11-26  Srikar Gupta  File header and function comments in the standard format.
"""

from unittest.mock import MagicMock

import numpy as np

from retrieval.vectorstore import PineconeStore


def _store(has_index=True):
    """
    Function name: _store
    Purpose: Build a PineconeStore (dimension 4, index "idx", namespace "ns") over a mocked
        client.
    Parameters:
        has_index (bool): Whether the mocked client reports that the index already exists.
    Returns:
        tuple[PineconeStore, MagicMock, MagicMock]: The store, the mocked client and the
            mocked index.
    """
    client = MagicMock()
    client.has_index.return_value = has_index
    store = PineconeStore(dim=4, index_name="idx", namespace="ns", client=client)
    return store, client, client.Index.return_value


def test_creates_index_when_missing():
    """
    Function name: test_creates_index_when_missing
    Purpose: Check that a missing index is created with the right name, dimension and cosine
        metric.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    _, client, _ = _store(has_index=False)
    kwargs = client.create_index.call_args.kwargs
    assert kwargs["name"] == "idx" and kwargs["dimension"] == 4 and kwargs["metric"] == "cosine"


def test_existing_index_not_recreated():
    """
    Function name: test_existing_index_not_recreated
    Purpose: Check that an existing index is not created again.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    _, client, _ = _store(has_index=True)
    client.create_index.assert_not_called()


def test_upsert_batches_with_metadata(chunks):
    """
    Function name: test_upsert_batches_with_metadata
    Purpose: Check that vectors are upserted in batches into the namespace with the chunk id,
        text and a page of -1 in place of None.
    Parameters:
        chunks (list[Chunk]): Test corpus fixture.
    Returns:
        None: The test passes or fails through assertions.
    """
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
    """
    Function name: test_search_maps_matches_back_to_chunks
    Purpose: Check that search queries the namespace with top_k and maps matches back to
        chunks with dense scores, turning page -1 back into None.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
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
    """
    Function name: test_count_reads_namespace_stats
    Purpose: Check that count() returns the vector count of the store's namespace from the
        index statistics.
    Parameters:
        None
    Returns:
        None: The test passes or fails through assertions.
    """
    store, _, index = _store()
    index.describe_index_stats.return_value = {"namespaces": {"ns": {"vector_count": 7}}}
    assert store.count() == 7
