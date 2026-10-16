import json
import sys
from pathlib import Path

import pytest
from haystack import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import make_eval                                                          # noqa: E402
from rag_agent.metrics import score                                       # noqa: E402

BY_ID = {"a/x": Document(content="Spend was 6.2 million dollars on the campaign.", meta={"chunk_id": "a/x"}),
         "b/y": Document(content="Retail media costs are rising every month.", meta={"chunk_id": "b/y"}),
         "c/z": Document(content="Creators performed best when they tried the product.", meta={"chunk_id": "c/z"})}


def item(*pairs, q="Q?"):
    return {"question": q, "reference_answer": "A.", "evidence": [{"chunk_id": c, "quote": s} for c, s in pairs]}


def test_validation_accepts_a_good_lookup_and_rejects_bad_quotes_and_ids():
    assert make_eval.validate(item(("a/x", "6.2 million dollars")), "lookup", BY_ID) is None
    assert make_eval.validate(item(("a/x", "7 million dollars")), "lookup", BY_ID) == "quote not found in chunk"
    assert make_eval.validate(item(("q/q", "6.2 million dollars")), "lookup", BY_ID) == "unknown chunk id"
    assert make_eval.validate({"question": "", "reference_answer": "x", "evidence": []}, "lookup", BY_ID) == "no question"


def test_validation_enforces_the_number_of_chunks_for_each_kind():
    two = item(("a/x", "6.2 million dollars"), ("b/y", "Retail media costs"))
    assert make_eval.validate(two, "lookup", BY_ID) == "needs exactly 1 chunk"
    assert make_eval.validate(two, "multi_hop", BY_ID) is None
    assert make_eval.validate(two, "broad", BY_ID) == "needs 3+ chunks"
    same_doc = {"a/x": BY_ID["a/x"], "a/w": Document(content="Another section of the same document here.", meta={"chunk_id": "a/w"})}
    assert make_eval.validate(item(("a/x", "6.2 million dollars"), ("a/w", "Another section")), "multi_hop", same_doc) == "needs 2+ chunks from 2+ documents"


def test_generate_keeps_only_valid_items_from_claude(fake_claude, monkeypatch):
    import fake_claude as fc
    good = item(("a/x", "6.2 million dollars"))
    bad = item(("a/x", "not in the chunk at all"))
    monkeypatch.setattr(fc, "reply_for", lambda system, user: json.dumps([good, bad]))
    chunks = list(BY_ID.values())
    out, dropped = make_eval.generate(2, chunks)
    assert [o["kind"] for o in out] == ["lookup", "context"] and dropped["quote not found in chunk"] == 4


def test_scoring_counts_ids_members_and_bigger_pieces():
    by_id = BY_ID
    assert score([by_id["a/x"]], ["a/x"], by_id) == {"hit": 1.0, "recall": 1.0, "rr": 1.0}
    summary = Document(content="short", meta={"chunk_id": "raptor/L1-1", "members": ["b/y"]})
    assert score([by_id["a/x"], summary], ["b/y"], by_id)["rr"] == 0.5
    parent = Document(content="Spend was 6.2 million dollars on the campaign. And more words follow.", meta={"chunk_id": "a/part"})
    assert score([parent], ["a/x"], by_id)["hit"] == 1.0
    assert score([by_id["c/z"]], ["a/x", "b/y"], by_id) == {"hit": 0.0, "recall": 0.0, "rr": 0.0}


def test_smoke_questions_are_mechanical_one_per_chunk(offline_index):
    qs = make_eval.smoke(offline_index.chunks)
    assert len(qs) == len(offline_index.chunks) and all(len(q["gold"]) == 1 for q in qs)
