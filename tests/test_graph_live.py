"""Runs ONLY when NEO4J_URI is set (CI starts Neo4j with the GDS plugin). This is the first real test of the Cypher and GDS calls."""
import os

import pytest

from rag_agent.graph_store import Neo4jStore

pytestmark = pytest.mark.skipif(not os.environ.get("NEO4J_URI"), reason="no live Neo4j (set NEO4J_URI to run)")


@pytest.fixture
def store():
    s = Neo4jStore.connect(os.environ["NEO4J_URI"], os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    s.load({"a": ["c1", "c2"], "b": ["c1"], "c": ["c2", "c3"], "d": ["c3"], "x": ["c4"], "y": ["c4"]},
           {("a", "b"): 1, ("a", "c"): 1, ("c", "d"): 2, ("x", "y"): 3})
    yield s
    s.close()


def test_roundtrip(store):
    assert store.entity_names() == ["a", "b", "c", "d", "x", "y"] and store.mentions_of(["a"]) == {"a": ["c1", "c2"]}


def test_gds_is_installed_and_pagerank_and_leiden_run(store):
    assert store.gds_version(), "install the Graph Data Science plugin"
    r = store.pagerank(["a"])
    assert r["a"] == max(r.values())
    groups = store.communities()
    assert any(set(g) >= {"x", "y"} for g in groups) and not any("x" in g and "a" in g for g in groups)


def test_the_projection_is_dropped_after_use(store):
    store.pagerank(["a"])
    rows = store._run("CALL gds.graph.list() YIELD graphName RETURN graphName")
    assert not [r for r in rows if r["graphName"].startswith("rag-")]
