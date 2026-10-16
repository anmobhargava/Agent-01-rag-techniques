"""Tests for the ideas, offline (no key). Each test says in its name what concept it protects."""
from rag_agent.index import load_chunks
from rag_agent.techniques import TECHNIQUES


def ids(docs):
    return [d.meta["chunk_id"] for d in docs]


def test_chunks_have_unique_ids_and_do_not_contain_the_document_title(offline_index):
    got = ids(offline_index.chunks)
    assert len(got) == len(set(got)) > 30
    results = offline_index.by_id["campaign-leaf-local/results"]
    assert "Leaf Local" not in results.content          # the whole point: this chunk does not name its campaign


def test_every_technique_returns_documents_with_ids_offline(offline_index):
    for key, t in TECHNIQUES.items():
        r = t.fn(offline_index, "Which agency ran Leaf Local and what is its fee?", 4)
        assert r.technique == key
        assert r.notes, key                                       # each one explains itself
        assert all(d.meta.get("chunk_id") for d in r.documents), key


def test_contextual_retrieval_finds_the_chunk_that_never_names_its_subject(offline_index):
    q = "How much did Crunch Time spend?"
    target = "campaign-crunch-time/results"
    assert target in ids(TECHNIQUES["contextual"].fn(offline_index, q, 3).documents)
    assert target not in ids(TECHNIQUES["naive"].fn(offline_index, q, 3).documents)


def test_hybrid_fuses_two_rankings_not_just_one(offline_index):
    q = "paid search saturated"
    hybrid = set(ids(TECHNIQUES["hybrid"].fn(offline_index, q, 5).documents))
    bm25 = set(ids(offline_index.plain.bm25(q, 5)))
    dense = set(ids(offline_index.plain.dense(q, 5)))
    assert hybrid <= bm25 | dense and hybrid & bm25 and hybrid & dense


def test_parent_document_returns_a_bigger_piece_than_the_one_it_matched(offline_index):
    leaves, merger = offline_index.parents
    hits = leaves.hybrid("Which agency does paid search?", 8)
    merged = merger.run(hits)["documents"]
    assert any(d.meta["__level"] < hits[0].meta["__level"] for d in merged)      # merged up a level


def test_raptor_tree_has_summary_levels_that_cover_every_chunk_once(offline_index):
    nodes = list(offline_index.raptor.by_id.values())
    level1 = [n for n in nodes if n.meta["level"] == 1]
    assert level1 and any(n.meta["level"] >= 2 for n in nodes)
    covered = [m for n in level1 for m in n.meta["members"]]
    assert sorted(covered) == sorted(ids(offline_index.chunks))                   # each chunk in exactly one cluster


def test_graph_seeds_use_the_longest_name_and_do_not_double_count(offline_index):
    g = offline_index.graph
    assert g.seeds("What happened with Leaf Crisps?") == ["leaf crisps"]          # not also 'leaf'
    assert g.seeds("Which agency ran Leaf Local?") == ["leaf local"]


def test_personalised_pagerank_is_a_probability_and_favours_the_seed_neighbourhood(offline_index):
    g = offline_index.graph
    r = g.store.pagerank(["leaf local"])
    assert abs(sum(r.values()) - 1) < 1e-6
    assert r["leaf local"] == max(r.values()) or r["leaf local"] > sum(r.values()) / len(r)


def test_graph_local_returns_chunks_that_mention_top_entities_and_global_returns_reports(offline_index):
    g = offline_index.graph
    docs, seeds, top = g.local("Tell me about Tidewater Creative", 4)
    assert docs and seeds == ["tidewater creative"]
    assert g.reports and g.global_("lessons about shoppers", 2)


def test_corrective_returns_nothing_for_a_question_the_documents_cannot_answer(offline_index):
    r = TECHNIQUES["corrective"].fn(offline_index, "What is the CEO's favourite colour?", 4)
    assert r.documents == [] and any("Still nothing" in n for n in r.notes)


def test_decompose_splits_a_two_part_question_offline():
    from rag_agent.components import QueryExpander
    out = QueryExpander("decompose").run("Which agency ran Leaf Local and what is its fee?")
    assert len(out["queries"]) == 2 and out["was_standin"]
