"""
Where the entity graph lives. Two backends with the same methods:

  MemoryStore   a Python dict. Default, so the repo runs with nothing installed. Small pure-Python PageRank and grouping.
  Neo4jStore    the real graph database. The two graph algorithms run INSIDE Neo4j using its Graph Data Science (GDS) library.

Pick with GRAPH_BACKEND=memory (default) or GRAPH_BACKEND=neo4j (then NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, optional NEO4J_DATABASE).

What the graph holds (this is GraphRAG built from the chunks, so there are no typed relations, only "appear together"):

    (:Entity {name: "leaf local"}) -[:MENTIONED_IN]-> (:Chunk {id: "campaign-leaf-local/results"})
    (:Entity {name: "leaf local"}) -[:CO_OCCURS {weight: 3}]-> (:Entity {name: "tidewater creative"})

An Entity is a dot. CO_OCCURS says two dots showed up in the same chunk, and weight is how many chunks they share.
Every node I write carries project: "rag-technique-agent", so this code only ever reads or deletes its own data, even in a
database that holds other things.

Neo4j needs the GDS plugin (Neo4j Desktop: your database > Plugins > Graph Data Science Library > Install; docker-compose.yml
installs it for the Docker route; AuraDB Free does not have it).

I have NOT run this against a live Neo4j yet. The Neo4j code is tested with a fake driver that checks the queries I
send, the safety rules and the parsing. The GDS calls (gds.graph.project, gds.pageRank.stream, gds.leiden.stream) are
written from the GDS documentation and have never executed. tests/test_graph_live.py runs them for real
when NEO4J_URI is set, and that is the first real test of this file.
"""

import os
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager

PROJECT = "rag-technique-agent"


class GraphEmpty(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------ memory

def _adjacency(edges: dict[tuple[str, str], float]) -> dict[str, dict[str, float]]:
    adj: dict[str, dict[str, float]] = defaultdict(dict)
    for (a, b), w in edges.items():
        if a != b:
            adj[a][b] = adj[a].get(b, 0) + w
            adj[b][a] = adj[b].get(a, 0) + w
    return adj


def _pagerank(edges, seeds, alpha: float = 0.85, tol: float = 1e-10) -> dict[str, float]:
    """Personalised PageRank by repeated passing of score along links. Each round every dot hands alpha of its score to its
    neighbours in proportion to the link weights, and (1 - alpha) of the total goes back to the seeds. Repeat until the scores stop
    moving. That is a walker who keeps jumping back to the start, written out. Neo4j's gds.pageRank does this at scale."""
    adj = _adjacency(edges)
    home = {n: (1 / len(seeds) if n in seeds else 0.0) for n in adj}
    score = dict(home)
    for _ in range(200):
        new = {n: (1 - alpha) * home[n] for n in adj}
        for n, friends in adj.items():
            total = sum(friends.values())
            for m, w in friends.items():
                new[m] += alpha * score[n] * w / total
        moved = sum(abs(new[n] - score[n]) for n in adj)
        score = new
        if moved < tol:
            break
    return score


def _communities(edges) -> list[list[str]]:
    """Groups of dots linked tightly to each other: the first pass of Louvain, in plain Python.
    Every dot starts alone. Visit the dots in alphabetical order and move each into the neighbouring group that most improves
    modularity (more link weight inside the group than chance would give). Repeat until nobody moves. Neo4j's Leiden adds more
    passes and fixes some bad merges; this is enough to show the idea and to test with."""
    adj = _adjacency(edges)
    degree = {n: sum(f.values()) for n, f in adj.items()}
    two_m = sum(degree.values())
    group = {n: n for n in adj}
    total = dict(degree)
    for _ in range(50):
        moved = False
        for n in sorted(adj):
            old = group[n]
            total[old] -= degree[n]
            links: Counter = Counter()
            for m, w in adj[n].items():
                links[group[m]] += w
            best, best_gain = old, links.get(old, 0) - total[old] * degree[n] / two_m
            for g in sorted(links):
                gain = links[g] - total[g] * degree[n] / two_m
                if gain > best_gain + 1e-12:
                    best, best_gain = g, gain
            total[best] += degree[n]
            group[n] = best
            moved |= best != old
        if not moved:
            break
    out: dict[str, list[str]] = defaultdict(list)
    for n, g in group.items():
        out[g].append(n)
    return sorted((sorted(v) for v in out.values()), key=lambda p: p[0])


class MemoryStore:
    kind = "memory"

    def __init__(self):
        self._mentions: dict[str, list[str]] = {}
        self._edges: dict[tuple[str, str], float] = {}
        self._reports: list[dict] = []

    def load(self, mentions: dict[str, list[str]], edges: dict[tuple[str, str], float]) -> None:
        self._mentions = {e: sorted(c) for e, c in mentions.items()}
        self._edges = dict(edges)
        self._reports = []

    def entity_names(self) -> list[str]:
        return sorted(self._mentions)

    def mentions_of(self, names) -> dict[str, list[str]]:
        return {n: list(self._mentions[n]) for n in names if n in self._mentions}

    def pagerank(self, seeds) -> dict[str, float]:
        return _pagerank(self._edges, set(seeds))

    def communities(self) -> list[list[str]]:
        return _communities(self._edges)

    def gds_version(self):
        return None

    def save_reports(self, reports: list[dict]) -> None:
        self._reports = reports

    def reports(self) -> list[dict]:
        return list(self._reports)

    def is_empty(self) -> bool:
        return not self._mentions

    def close(self) -> None:
        pass


# ------------------------------------------------------------------------------------------ neo4j

class Neo4jStore:
    kind = "neo4j"

    def __init__(self, driver, database: str | None = None, project: str = PROJECT):
        self.driver, self.database, self.project = driver, database, project

    @classmethod
    def connect(cls, uri: str, user: str, password: str, database: str | None = None) -> "Neo4jStore":
        from neo4j import GraphDatabase          # imported here so the memory backend needs no neo4j package
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()             # fail now with a clear error, not on the first question
        return cls(driver, database or None)

    def _run(self, query: str, **params):
        res = self.driver.execute_query(query, {"p": self.project, **params}, database_=self.database)
        return res.records

    # ---- writing (once, when the index is built)

    def load(self, mentions: dict[str, list[str]], edges: dict[tuple[str, str], float]) -> None:
        """Replace this project's graph. Names and ids always go in as PARAMETERS, never into the query text."""
        self._run("CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (n:Entity) REQUIRE (n.project, n.name) IS UNIQUE")
        self._run("CREATE CONSTRAINT chunk_key IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.project, c.id) IS UNIQUE")
        self._run("MATCH (n) WHERE (n:Entity OR n:Chunk OR n:Report) AND n.project = $p DETACH DELETE n")
        self._run("UNWIND $names AS name MERGE (:Entity {project: $p, name: name})", names=sorted(mentions))
        chunk_ids = sorted({c for cs in mentions.values() for c in cs})
        self._run("UNWIND $ids AS id MERGE (:Chunk {project: $p, id: id})", ids=chunk_ids)
        self._run("UNWIND $rows AS row "
                  "MATCH (e:Entity {project: $p, name: row.e}), (c:Chunk {project: $p, id: row.c}) "
                  "MERGE (e)-[:MENTIONED_IN]->(c)",
                  rows=[{"e": e, "c": c} for e, cs in sorted(mentions.items()) for c in cs])
        self._run("UNWIND $rows AS row "
                  "MATCH (a:Entity {project: $p, name: row.a}), (b:Entity {project: $p, name: row.b}) "
                  "MERGE (a)-[r:CO_OCCURS]->(b) SET r.weight = row.w",
                  rows=[{"a": a, "b": b, "w": w} for (a, b), w in sorted(edges.items())])

    def save_reports(self, reports: list[dict]) -> None:
        self._run("MATCH (r:Report {project: $p}) DETACH DELETE r")
        self._run("UNWIND $rows AS row CREATE (:Report {project: $p, id: row.id, members: row.members, text: row.text})",
                  rows=reports)

    # ---- reading (every question)

    def entity_names(self) -> list[str]:
        return [r["name"] for r in self._run("MATCH (n:Entity {project: $p}) RETURN n.name AS name ORDER BY name")]

    def mentions_of(self, names) -> dict[str, list[str]]:
        rows = self._run("MATCH (e:Entity {project: $p})-[:MENTIONED_IN]->(c:Chunk {project: $p}) WHERE e.name IN $names "
                         "RETURN e.name AS name, collect(c.id) AS chunks ORDER BY name", names=list(names))
        return {r["name"]: sorted(r["chunks"]) for r in rows}

    # ---- the graph algorithms, run inside Neo4j by the Graph Data Science library

    def _gds(self, query: str, **params):
        try:
            return self._run(query, **params)
        except Exception as e:
            text = str(e).lower()
            if "gds" in text and ("procedure" in text or "function" in text):
                raise RuntimeError("Neo4j does not have the Graph Data Science library. In Neo4j Desktop: select the database > "
                                   "Plugins > Graph Data Science Library > Install, then restart it. (docker-compose.yml installs it "
                                   "for the Docker route. AuraDB Free does not offer it.)") from e
            raise

    def gds_version(self):
        try:
            return self._gds("RETURN gds.version() AS v")[0]["v"]
        except RuntimeError:
            return None

    @contextmanager
    def _projection(self):
        """GDS runs on an in-memory COPY of the graph called a projection, not on the database directly. I project just this
        project's entities and their weighted CO_OCCURS links (treated as undirected), use it, and always drop it afterwards
        so nothing piles up in Neo4j's memory."""
        name = f"rag-{uuid.uuid4().hex[:10]}"
        self._gds("MATCH (a:Entity {project: $p})-[r:CO_OCCURS]->(b:Entity {project: $p}) "
                  "WITH gds.graph.project($g, a, b, {relationshipType: 'CO_OCCURS', relationshipProperties: r {.weight}}, "
                  "{undirectedRelationshipTypes: ['CO_OCCURS']}) AS graph "
                  "RETURN graph.graphName AS name", g=name)
        try:
            yield name
        finally:
            self._run("CALL gds.graph.drop($g, false)", g=name)

    def pagerank(self, seeds) -> dict[str, float]:
        """Personalised PageRank: sourceNodes are the seeds, so the walker keeps jumping back to them. Link weights count."""
        with self._projection() as g:
            rows = self._gds("MATCH (n:Entity {project: $p}) WHERE n.name IN $seeds WITH collect(n) AS seedNodes "
                             "CALL gds.pageRank.stream($g, {sourceNodes: seedNodes, relationshipWeightProperty: 'weight', "
                             "dampingFactor: 0.85, maxIterations: 100, tolerance: 1e-7}) "
                             "YIELD nodeId, score RETURN gds.util.asNode(nodeId).name AS name, score", g=g, seeds=list(seeds))
        return {r["name"]: r["score"] for r in rows}

    def communities(self) -> list[list[str]]:
        """Leiden, an improved Louvain. randomSeed and one thread, so the same graph gives the same groups."""
        with self._projection() as g:
            rows = self._gds("CALL gds.leiden.stream($g, {randomSeed: 42, concurrency: 1, relationshipWeightProperty: 'weight'}) "
                             "YIELD nodeId, communityId RETURN gds.util.asNode(nodeId).name AS name, communityId", g=g)
        out: dict[int, list[str]] = defaultdict(list)
        for r in rows:
            out[r["communityId"]].append(r["name"])
        return sorted((sorted(v) for v in out.values()), key=lambda p: p[0])

    def reports(self) -> list[dict]:
        rows = self._run("MATCH (r:Report {project: $p}) RETURN r.id AS id, r.members AS members, r.text AS text ORDER BY id")
        return [{"id": r["id"], "members": list(r["members"]), "text": r["text"]} for r in rows]

    def is_empty(self) -> bool:
        return not self._run("MATCH (n:Entity {project: $p}) RETURN n LIMIT 1")

    def close(self) -> None:
        self.driver.close()


# ------------------------------------------------------------------------------------------ choosing

def get_store():
    """GRAPH_BACKEND=memory (default) or neo4j. If neo4j is asked for and NEO4J_URI is missing, stop with a message instead of
    quietly running on the in-memory stand-in and letting anyone believe the database was involved."""
    backend = os.environ.get("GRAPH_BACKEND", "memory")
    if backend == "memory":
        return MemoryStore()
    if backend != "neo4j":
        raise ValueError(f"GRAPH_BACKEND must be 'memory' or 'neo4j', got {backend!r}")
    uri = os.environ.get("NEO4J_URI")
    if not uri:
        raise RuntimeError("GRAPH_BACKEND=neo4j but NEO4J_URI is not set. Start Neo4j with `docker compose up -d neo4j` and set "
                           "NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=<password in docker-compose.yml>. "
                           "To run without a database, unset GRAPH_BACKEND.")
    return Neo4jStore.connect(uri, os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""),
                              os.environ.get("NEO4J_DATABASE"))
