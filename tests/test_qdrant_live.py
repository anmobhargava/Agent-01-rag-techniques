"""Runs ONLY when QDRANT_URL is set (CI starts a Qdrant server): the same index, but against a real server instead of in-process."""
import os

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("QDRANT_URL"), reason="no Qdrant server (set QDRANT_URL to run)")


def test_hybrid_and_filters_and_multivector_work_on_a_server():
    from src.rag_agent.index import Index
    ix = Index()
    assert ix.plain.hybrid("How much did Crunch Time spend?", 3)
    f = {"field": "meta.doc_type", "operator": "==", "value": "research"}
    assert all(d.meta["doc_type"] == "research" for d in ix.plain.hybrid("price", 3, filters=f))
    assert ix.colbert.search("Crunch Time spend", 3)


def test_server_collection_names_are_stable_for_the_same_corpus():
    from src.rag_agent.index import collection_name
    assert collection_name("plain", ["a", "b"], 32) == collection_name("plain", ["a", "b"], 32)
    assert collection_name("plain", ["a", "b"], 32) != collection_name("plain", ["a", "c"], 32)
