"""Semantic cache (the guards matter more than the hit), adaptive retrieval, and the Self-RAG-style self-check."""
import numpy as np
import pytest

from rag_agent import agent
from rag_agent.agent import Answer
from rag_agent.semantic_cache import SemanticCache, key_facts


class LookalikeEmbedder:
    """Every question maps to the SAME vector: a worst case where the neural model thinks all questions are one question."""
    real, dims, name = True, 4, "all-the-same"

    def query(self, text):
        return np.array([1.0, 0, 0, 0])

    def docs(self, texts):
        return np.tile([1.0, 0, 0, 0], (len(texts), 1))


def make_answer(status="ok", unknown=None, text="6.2 million [campaign-crunch-time/results]"):
    return Answer("q", "q", status, text, "hybrid", "why", [], [], unknown or [], live=False)


def test_key_facts_pull_out_names_numbers_and_periods():
    assert key_facts("How much did Crunch Time spend in Q3 2026 on 2.6 million?") == ["2.6", "2026", "crunch time", "q3"]
    assert key_facts("what is the fee") == []


def test_cache_hit_on_the_same_question_and_miss_when_the_names_differ_even_with_identical_vectors():
    c = SemanticCache(LookalikeEmbedder(), "corpus-1")
    assert c.put("How much did Crunch Time spend?", make_answer(), live=False)
    hit = c.get("How much did Crunch Time spend?", live=False)
    assert hit and hit[0]["text"].startswith("6.2")
    assert c.get("How much did Leaf Local spend?", live=False) is None            # similarity is 1.0 here; the facts guard is what blocks it


def test_cache_does_not_mix_live_with_offline_or_corpora_or_expired_entries(monkeypatch):
    c = SemanticCache(LookalikeEmbedder(), "corpus-1", ttl=10)
    c.put("How much did Crunch Time spend?", make_answer(), live=False)
    assert c.get("How much did Crunch Time spend?", live=True) is None
    other = SemanticCache(LookalikeEmbedder(), "corpus-2")
    assert other.get("How much did Crunch Time spend?", live=False) is None
    import time
    real = time.time()
    monkeypatch.setattr("rag_agent.semantic_cache.time.time", lambda: real + 11)
    assert c.get("How much did Crunch Time spend?", live=False) is None


def test_cache_refuses_refusals_blocked_and_unverified_answers():
    c = SemanticCache(LookalikeEmbedder(), "corpus-1")
    for bad in (make_answer("no_evidence"), make_answer("blocked"), make_answer("ok", unknown=["x/y"])):
        assert not c.put("q", bad, live=False)
    assert c.stats()["entries"] == 0


def test_cache_evicts_the_oldest_entry_past_max_size():
    c = SemanticCache(LookalikeEmbedder(), "corpus-1", max_size=2)
    for name in ("Alpha Beta", "Gamma Delta", "Omega Sigma"):
        c.put(f"What about {name}?", make_answer(), live=False)
    assert c.stats()["entries"] == 2
    assert c.get("What about Alpha Beta?", live=False) is None and c.get("What about Omega Sigma?", live=False)


def test_threshold_is_stricter_for_the_stand_in(offline_index):
    assert SemanticCache(offline_index.embedder, "h").threshold == pytest.approx(0.97)
    assert SemanticCache(LookalikeEmbedder(), "h").threshold == pytest.approx(0.92)


def test_agent_uses_the_cache_for_a_repeat_and_skips_it_when_a_technique_is_forced(offline_index):
    cache = SemanticCache(offline_index.embedder, offline_index.corpus_hash)
    q = "How much did Crunch Time spend?"
    first = agent.run(q, offline_index, cache=cache)
    assert not first.cached
    events = []
    second = agent.run(q, offline_index, cache=cache, emit=lambda k, p: events.append(k))
    assert second.cached and second.cost_usd == 0 and second.text == first.text and "token" in events
    assert not agent.run(q, offline_index, cache=cache, forced="hybrid").cached


def test_adaptive_levels_offline():
    assert agent.adaptive_level("hello")[0] == "none"
    assert agent.adaptive_level("How much did Crunch Time spend?")[0] == "single"
    assert agent.adaptive_level("Which agency ran the campaign that had the best return?")[0] == "multi"


def test_small_talk_costs_nothing_and_searches_nothing(offline_index, fake_claude):
    import fake_claude as fc
    fc.OVERRIDES["decide how much retrieval"] = '{"level": "none", "why": "greeting"}'
    try:
        a = agent.run("hi there friend", offline_index)
    finally:
        fc.OVERRIDES.clear()
    assert a.status == "no_retrieval" and a.attempts == []
    assert not any("You answer questions" in c for c in fake_claude.calls)


def test_multi_level_makes_the_planner_pick_a_multi_step_technique(offline_index):
    got, _ = agent.plan("Which agency ran the campaign that had the best return?", offline_index, "multi")
    assert got in agent.MULTI_STEP


def test_offline_critic_flags_a_sentence_the_chunks_do_not_support(offline_index):
    ev = [offline_index.by_id["campaign-crunch-time/results"]]
    bad = agent.critique("How much did Crunch Time spend?", "Crunch Time spent nine billion euros on submarine advertising.", ev)
    good = agent.critique("How much did Crunch Time spend?", agent.extractive_answer("How much did Crunch Time spend?", ev), ev)
    assert not bad["supported"] and bad["standin"] and good["supported"]


def test_live_self_check_rewrites_once_and_emits_a_replace_event(offline_index, fake_claude):
    import fake_claude as fc
    fc.OVERRIDES["check an answer against the context"] = '{"supported": false, "unsupported": ["made up"], "useful": true}'
    events = []
    try:
        a = agent.run("How much did Crunch Time spend?", offline_index, emit=lambda k, p: events.append(k))
    finally:
        fc.OVERRIDES.clear()
    assert a.revised and "replace" in events and events.count("replace") == 1
    assert any("Your previous answer contained claims" in c for c in fake_claude.systems)
