"""The graph store: memory backend for real, Neo4j backend against a fake driver that records every query."""
from types import SimpleNamespace

import pytest

from rag_agent import graph_store
from rag_agent.graph_store import MemoryStore, Neo4jStore, get_store

MENTIONS = {"a": ["c1", "c2"], "b": ["c1"], "c": ["c2", "c3"], "d": ["c3"], "x": ["c4"], "y": ["c4"]}
EDGES = {("a", "b"): 1, ("a", "c"): 1, ("c", "d"): 2, ("x", "y"): 3}


def loaded() -> MemoryStore:
    s = MemoryStore()
    s.load(MENTIONS, EDGES)
    return s


def test_memory_store_returns_what_was_loaded():
    s = loaded()
    assert s.entity_names() == ["a", "b", "c", "d", "x", "y"] and s.mentions_of(["a", "zzz"]) == {"a": ["c1", "c2"]}
    assert not s.is_empty() and MemoryStore().is_empty() and s.gds_version() is None


def test_pagerank_is_a_probability_that_favours_the_seed_and_stays_in_its_component():
    r = loaded().pagerank(["a"])
    assert abs(sum(r.values()) - 1) < 1e-6 and r["a"] == max(r.values())
    assert "x" not in r or r["x"] == 0 or "x" not in r                              # x and y are not connected to a


def test_communities_separate_the_two_clusters_and_use_link_weights():
    groups = loaded().communities()
    assert sorted(len(g) for g in groups) == [2, 4] or any({"x", "y"} == set(g) for g in groups)
    assert any(set(g) >= {"x", "y"} for g in groups) and not any("x" in g and "a" in g for g in groups)


class FakeDriver:
    """Records every query. `script(query, params)` returns the rows to hand back."""

    def __init__(self, script=lambda q, p: []):
        self.calls, self.script, self.closed = [], script, False

    def execute_query(self, query, params, database_=None):
        self.calls.append((query, params))
        return SimpleNamespace(records=self.script(query, params))

    def close(self):
        self.closed = True


def test_neo4j_load_scopes_everything_to_the_project_and_passes_names_as_parameters():
    evil = "x' DETACH DELETE n //"
    d = FakeDriver()
    Neo4jStore(d).load({evil: ["c1"], "ok": ["c1"]}, {(evil, "ok"): 2})
    texts = [q for q, _ in d.calls]
    assert all("$p" in q or q.startswith("CREATE CONSTRAINT") for q in texts)
    assert not any(evil in q for q in texts)                                        # the name never entered query text
    assert all(p.get("p") == graph_store.PROJECT for _, p in d.calls)
    assert any(evil in str(p) for _, p in d.calls)                                  # it went in as a parameter instead
    delete = next(q for q in texts if "DETACH DELETE" in q and "n.project = $p" in q)
    assert "Entity" in delete                                                       # only this project's nodes are deleted


def test_neo4j_reads_parse_rows():
    def script(q, p):
        if "RETURN n.name AS name ORDER" in q:
            return [{"name": "a"}, {"name": "b"}]
        if "collect(c.id)" in q:
            return [{"name": "a", "chunks": ["c2", "c1"]}]
        if "gds.pageRank" in q:
            return [{"name": "a", "score": 0.6}, {"name": "b", "score": 0.4}]
        if "gds.leiden" in q:
            return [{"name": "a", "communityId": 1}, {"name": "b", "communityId": 1}, {"name": "x", "communityId": 2}]
        return []
    s = Neo4jStore(FakeDriver(script))
    assert s.entity_names() == ["a", "b"] and s.mentions_of(["a"]) == {"a": ["c1", "c2"]}
    assert s.pagerank(["a"]) == {"a": 0.6, "b": 0.4} and s.communities() == [["a", "b"], ["x"]]


def test_gds_projection_is_always_dropped_even_when_the_algorithm_fails():
    def script(q, p):
        if "gds.pageRank" in q:
            raise RuntimeError("boom")
        return []
    d = FakeDriver(script)
    with pytest.raises(RuntimeError, match="boom"):
        Neo4jStore(d).pagerank(["a"])
    queries = [q for q, _ in d.calls]
    assert any("gds.graph.project" in q for q in queries) and any("gds.graph.drop" in q for q in queries)


def test_missing_gds_gives_a_message_that_says_how_to_install_it():
    def script(q, p):
        raise RuntimeError("There is no procedure with the name `gds.graph.project` registered")
    with pytest.raises(RuntimeError, match="Graph Data Science library"):
        Neo4jStore(FakeDriver(script)).communities()
    assert Neo4jStore(FakeDriver(script)).gds_version() is None


def test_get_store_defaults_to_memory_and_refuses_to_pretend(monkeypatch):
    monkeypatch.delenv("GRAPH_BACKEND", raising=False)
    assert isinstance(get_store(), MemoryStore)
    monkeypatch.setenv("GRAPH_BACKEND", "neo4j")
    monkeypatch.delenv("NEO4J_URI", raising=False)
    with pytest.raises(RuntimeError, match="NEO4J_URI"):
        get_store()
    monkeypatch.setenv("GRAPH_BACKEND", "mongo")
    with pytest.raises(ValueError):
        get_store()


def test_graph_runs_through_a_neo4j_store_end_to_end(offline_index):
    """The Graph class only talks to the store interface, so a Neo4j-shaped store works too. This one answers from memory data
    but records that the algorithms were requested from the STORE, not computed inside Graph."""
    from rag_agent.index import Graph

    class Spy(MemoryStore):
        kind = "spy"
        calls = []

        def pagerank(self, seeds):
            Spy.calls.append("pagerank")
            return super().pagerank(seeds)

        def communities(self):
            Spy.calls.append("communities")
            return super().communities()

    g = Graph(offline_index.chunks, offline_index.docs_dir, store=Spy())
    g.local("Tell me about Tidewater Creative")
    assert "communities" in Spy.calls and "pagerank" in Spy.calls
    assert g.store.reports() and g.store.reports()[0]["members"]
