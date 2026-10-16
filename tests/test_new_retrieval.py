"""Qdrant-backed stacks, metadata filters, MaxSim in Qdrant, and the five newer techniques."""
import numpy as np

from src.rag_agent import techniques
from src.rag_agent.components import FilterExtractor, GapFinder
from src.rag_agent.embedders import maxsim
from src.rag_agent.index import MultiVectorIndex, doc_meta, full_text, split_sentences


def test_doc_meta_comes_from_the_file_name():
    assert doc_meta("campaign-leaf-local") == {"doc_type": "campaign", "period": "all year"}
    assert doc_meta("q3-2026-review") == {"doc_type": "review", "period": "Q3 2026"}
    assert doc_meta("research-gen-z")["doc_type"] == "research"


def test_split_sentences_loses_nothing():
    text = "Spend was 6.2 million. It rose 3.5% in Q3! Did it? Yes.\n\nNew paragraph here."
    assert "".join(split_sentences(text)) == text


def test_dense_search_in_qdrant_returns_ranked_chunks(offline_index):
    hits = offline_index.plain.dense("How much did Crunch Time spend?", 5)
    assert 1 <= len(hits) <= 5 and any("crunch-time" in d.meta["chunk_id"] for d in hits)


def test_metadata_filter_applies_to_dense_bm25_and_hybrid(offline_index):
    f = {"field": "meta.doc_type", "operator": "==", "value": "research"}
    for fn in (offline_index.plain.dense, offline_index.plain.bm25, offline_index.plain.hybrid):
        hits = fn("what do shoppers think about price", 6, filters=f)
        assert hits and all(d.meta["doc_type"] == "research" for d in hits), fn.__name__


def test_multivector_maxsim_in_qdrant_matches_numpy(offline_index):
    """Qdrant's MAX_SIM must give the same score as my own maxsim(), or the 'late interaction' label would be a lie."""
    mv = offline_index.colbert
    q = "Crunch Time spend"
    top = mv.search(q, 3)[0]
    qv = offline_index.embedder.tokens([q], is_query=True)[0]
    dv = offline_index.embedder.tokens([top.content])[0]
    assert np.isclose(top.score, maxsim(qv, dv), atol=1e-4)
    assert isinstance(mv, MultiVectorIndex)


def test_filter_extractor_offline_uses_rules_and_says_so(offline_index):
    fx = FilterExtractor(offline_index.values)
    out = fx.run(query="What does the research say about Q3 prices?")
    flat = str(out["filters"])
    assert "research" in flat and "Q3" in flat and out["was_standin"]
    assert FilterExtractor(offline_index.values).run(query="How much did Crunch Time spend?")["filters"] is None


def test_filter_extractor_live_only_accepts_values_that_exist(offline_index, fake_claude):
    import fake_claude as fc
    fc.OVERRIDES["turn a question into metadata filters"] = '{"doc_type": "poetry", "period": "Q2 2026"}'
    try:
        out = FilterExtractor(offline_index.values).run(query="what do the research reports say")
    finally:
        fc.OVERRIDES.clear()
    assert "poetry" not in str(out["filters"])                            # invented value dropped, valid one kept
    assert not out["was_standin"]


def test_gap_finder_offline_names_question_terms_no_chunk_contains(offline_index):
    docs = [offline_index.by_id["brand-guidelines/voice"]]
    out = GapFinder().run(question="What is the zebra budget for Leaf Local?", documents=docs)
    assert "zebra" in out["query"].lower() and out["was_standin"]


def run_all(ix, question):
    return {key: techniques.TECHNIQUES[key].fn(ix, question, 5) for key in techniques.TECHNIQUES}


def test_every_technique_returns_documents_with_chunk_ids_and_notes(offline_index):
    for key, r in run_all(offline_index, "How much did Crunch Time spend and which agency ran it?").items():
        assert r.documents, key
        assert all("chunk_id" in d.meta for d in r.documents), key
        assert r.notes and r.technique == key, key


def test_new_techniques_are_registered_with_a_reason_to_use_each():
    for key in ("sentence_window", "late_chunking", "colbert", "filtered", "iterative"):
        assert key in techniques.TECHNIQUES and techniques.TECHNIQUES[key].when


def test_sentence_window_returns_the_matched_sentence_plus_neighbours(offline_index):
    r = techniques.sentence_window(offline_index, "How much did Crunch Time spend?", 3)
    d = r.documents[0]
    assert len(d.content) >= len(max(split_sentences(d.content), key=len))
    assert "window" in " ".join(r.notes).lower() or "sentence" in " ".join(r.notes).lower()


def test_filtered_technique_only_returns_chunks_from_the_filtered_type(offline_index):
    r = techniques.filtered(offline_index, "What does the research say about price?", 5)
    assert r.documents and {d.meta["doc_type"] for d in r.documents} == {"research"}


def test_iterative_does_more_than_one_search_when_something_is_missing(offline_index):
    r = techniques.iterative(offline_index, "Which agency ran Leaf Local and what is the zebra fee?", 5, max_hops=3)
    assert len(r.notes) >= 2


def test_late_chunking_says_it_is_a_stand_in_without_a_real_model(offline_index):
    r = techniques.late_chunking(offline_index, "How much did Crunch Time spend?", 3)
    assert r.standins and any("toy" in s.lower() or "stand-in" in s.lower() for s in r.standins)
    assert full_text("campaign-crunch-time")                              # late chunking reads the whole file
