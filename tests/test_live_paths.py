"""
The code paths that only run with a Claude key, tested against a scripted fake client (tests/fake_claude.py).
These check my code (asks, parses, falls back). They do not check how good Claude is at the jobs.
"""
from rag_agent import llm, tracing
from rag_agent.components import (ClaudeReranker, HydeQuery, QueryExpander, RelevanceGrader)
from rag_agent.index import Index
from rag_agent.techniques import TECHNIQUES
from haystack import Document


def docs3():
    return [Document(content=f"chunk {i}", meta={"chunk_id": f"d/{i}"}) for i in range(3)]


def test_reranker_uses_claudes_order(fake_claude):
    out = ClaudeReranker().run(docs3(), "q", top_k=3)["documents"]
    assert [d.meta["chunk_id"] for d in out] == ["d/1", "d/0", "d/2"]


def test_hyde_searches_with_the_pretend_answer(fake_claude):
    out = HydeQuery().run("How much did Leaf Local cost?")
    assert "6.2 million" in out["text"] and not out["was_standin"]


def test_multi_query_and_decompose_parse_claudes_lists(fake_claude):
    assert len(QueryExpander("paraphrase").run("q")["queries"]) == 4              # original + 3 rewrites
    assert QueryExpander("decompose").run("q")["queries"][0].startswith("Which agency")


def test_grader_keeps_the_chunks_claude_names(fake_claude):
    out = RelevanceGrader().run("q", docs3())
    assert [d.meta["chunk_id"] for d in out["relevant"]] == ["d/0", "d/1"] and out["score"] == 0.667 and not out["was_standin"]


def test_garbage_replies_fall_back_instead_of_crashing(monkeypatch, fake_claude):
    import fake_claude as fc
    monkeypatch.setattr(fc, "reply_for", lambda system, user: "sorry, no idea")
    assert len(ClaudeReranker().run(docs3(), "word", top_k=2)["documents"]) == 2
    assert QueryExpander("decompose").run("Which agency ran it and what is its fee?")["was_standin"]
    assert RelevanceGrader().run("word", docs3())["was_standin"] is False        # '[]'-less reply parses to nothing relevant


def test_an_api_error_falls_back_and_is_recorded(monkeypatch, fake_claude):
    def boom(self, **kw):
        raise RuntimeError("api down")
    monkeypatch.setattr(fake_claude, "_create", boom)
    with tracing.request_trace("t", question="q") as tr:
        assert llm.ask("s", "u", "x", offline="fallback") == "fallback"
    assert tr.meta["errors"] == ["x: RuntimeError"]


def test_every_technique_runs_with_a_live_client_and_costs_are_traced(fake_claude):
    ix = Index()
    with tracing.request_trace("t", question="q") as tr:
        for t in TECHNIQUES.values():
            assert isinstance(t.fn(ix, "How much did Leaf Local cost?", 4).documents, list), t.key
    assert tr.cost_usd > 0 and {c["type"] for c in tr.llm} >= {"rerank", "hyde", "grade", "contextualize", "summarise"}


def test_streaming_answer_is_assembled_and_costed(fake_claude):
    with tracing.request_trace("t", question="q") as tr:
        text = "".join(llm.ask_stream("You answer questions about Crunchwell", "u", "answer"))
    assert "6.2 million" in text and tr.llm[-1]["tokens_in"] == 100


def test_a_stream_that_dies_midway_does_not_get_the_fallback_glued_on(monkeypatch, fake_claude):
    class Dies:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        @property
        def text_stream(self):
            yield "Half an answer "
            raise RuntimeError("connection dropped")

    monkeypatch.setattr(fake_claude, "_stream", lambda self, **kw: Dies())
    with tracing.request_trace("t", question="q") as tr:
        text = "".join(llm.ask_stream("s", "u", "answer", offline="FALLBACK"))
    assert text == "Half an answer " and tr.meta["errors"] == ["answer: RuntimeError"]


def test_a_stream_that_fails_before_any_text_uses_the_fallback(monkeypatch, fake_claude):
    def boom(self, **kw):
        raise RuntimeError("down")
    monkeypatch.setattr(fake_claude, "_stream", boom)
    assert "".join(llm.ask_stream("s", "u", "answer", offline="FALLBACK")) == "FALLBACK"
