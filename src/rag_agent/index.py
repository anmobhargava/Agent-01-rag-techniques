"""
Everything that gets built ONCE from the documents, before any question is asked.

  load_chunks        split each markdown file into one chunk per "## section", with metadata and character offsets
  Stack              dense search in Qdrant (a real vector database) + BM25 keywords, with the hybrid pipeline wired up
  MultiVectorIndex   ColBERT-style: one vector per TOKEN, stored in Qdrant as a multi-vector, scored with MaxSim
  Index              the plain stack, plus the heavier ones built on first use:
                       contextual       every chunk gets a sentence saying where it comes from
                       parents          small pieces to search, big pieces to return (parent-document)
                       sentences        search single sentences, return the sentences around them (sentence window)
                       late             embed the whole document first, then cut it into chunk vectors (late chunking)
                       colbert          one vector per token (late interaction)
                       raptor           summaries of summaries (a tree), searched all at once
                       graph            entities and which chunks mention them (GraphRAG)

The heavier builds can call Claude (a summary per cluster, a context line per chunk). Results are cached in data/cache so
I pay once. With no API key they use plain stand-ins and say so in `index.standins`.

Vector database: Qdrant. By default it runs inside this process (QDRANT location ":memory:"), which is a real Qdrant
with the same API. Set QDRANT_URL (and QDRANT_API_KEY) to use a Qdrant server instead, e.g. the one in docker-compose.
"""

import dataclasses
import hashlib
import json
import os
import re
import uuid
from functools import cached_property
from pathlib import Path

import numpy as np
from haystack import Document, Pipeline
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentSplitter, HierarchicalDocumentSplitter
from haystack.components.retrievers import AutoMergingRetriever, InMemoryBM25Retriever, SentenceWindowRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from qdrant_client import QdrantClient, models
from sklearn.cluster import KMeans

from .components import DocumentEmbedder, TextEmbedder, _json_list
from .embedders import HFLateChunker, LsaEmbedder, ToyContextEncoder, get_embedder
from .graph_store import get_store
from .llm import STRONG, ask, have_key

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "data" / "docs"
CACHE_DIR = ROOT / "data" / "cache"


# ------------------------------------------------------------------------------------------ chunking

def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


DOC_TYPES = ("campaign", "research", "review", "scorecard", "playbook", "guidelines", "memo", "overview", "readout")


def doc_meta(stem: str) -> dict:
    """Metadata worked out from the FILE NAME by a rule, so I do not type it per document.
    'campaign-leaf-local' -> doc_type campaign;  'q3-2026-review' -> doc_type review, period 'Q3 2026'."""
    words = stem.split("-")
    doc_type = next((t for t in DOC_TYPES if t in words), "other")
    m = re.search(r"\b(h[12]|q[1-4])-(20\d\d)\b", stem)
    period = f"{m.group(1).upper()} {m.group(2)}" if m else ("FY2026" if "2026" in stem else "all year")
    return {"doc_type": doc_type, "period": period}


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """One chunk per '## section'. The chunk text is the section heading plus its body and nothing else.
    I leave the document title OUT on purpose: a 'Results' chunk then does not say which campaign it is about,
    which is exactly the real-world problem that contextual retrieval exists to fix.
    Each chunk also records where it sits in the file (char_start, char_end), which late chunking needs."""
    chunks = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        text = path.read_text()
        title = re.search(r"^# (.+)$", text, re.M).group(1).strip()
        heads = list(re.finditer(r"^## (.+)$", text, re.M))
        title_end = re.search(r"^# .+$", text, re.M).end()
        pieces = [("intro", title_end, heads[0].start() if heads else len(text), text[title_end:heads[0].start() if heads else len(text)])]
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            pieces.append((h.group(1).strip(), h.start(), end, text[h.end():end]))
        for heading, start, end, body in pieces:
            body = body.strip()
            if not body:
                continue
            cid = f"{path.stem}/{slug(heading)}"
            content = body if heading == "intro" else f"{heading}\n{body}"
            chunks.append(Document(id=cid, content=content,
                                   meta={"chunk_id": cid, "doc_id": path.stem, "title": title, "section": heading,
                                         "char_start": start, "char_end": end, **doc_meta(path.stem)}))
    return chunks


def split_sentences(text: str) -> list[str]:
    """Cut text into sentences WITHOUT losing a single character (pieces join back into the original), so the character
    offsets Haystack records for each piece stay correct. A blank piece is glued onto the one before it."""
    pieces: list[str] = []
    for piece in re.split(r"(?<=[.!?])(?=\s)|(?<=\n)", text):
        if pieces and not piece.strip():
            pieces[-1] += piece
        elif piece:
            pieces.append(piece)
    return pieces


def full_text(doc_id: str, docs_dir: Path = DOCS_DIR) -> str:
    return (Path(docs_dir) / f"{doc_id}.md").read_text()


# --------------------------------------------------------------------------------------------- cache

def cached(name: str, key: str, make):
    """Tiny JSON cache so a model-written thing is only paid for once. Offline stand-ins are cheap and my rules
    change while I work on them, so they are never cached. Only real Claude output is."""
    if not have_key():
        return make()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.json"
    try:
        store = json.loads(path.read_text()) if path.exists() else {}
    except json.JSONDecodeError:
        store = {}
    h = hashlib.sha1(key.encode()).hexdigest()
    if h not in store:
        store[h] = make()
        path.write_text(json.dumps(store, indent=1))
    return store[h]


def mode() -> str:
    return "live" if have_key() else "offline"


# ------------------------------------------------------------------------------------------- Qdrant

def qdrant_kwargs() -> dict:
    """In-process Qdrant by default; a Qdrant server when QDRANT_URL is set."""
    if os.environ.get("QDRANT_URL"):
        return {"url": os.environ["QDRANT_URL"], "api_key": Secret.from_env_var("QDRANT_API_KEY", strict=False)}
    return {"location": ":memory:"}


def collection_name(label: str, texts: list[str], dims: int) -> str:
    """Server mode: the same corpus and model always map to the same collection name. In-process: just be unique."""
    if os.environ.get("QDRANT_URL"):
        return f"{label}-{dims}-{hashlib.sha1('|'.join(texts).encode()).hexdigest()[:8]}"
    return f"{label}-{uuid.uuid4().hex[:8]}"


def qdrant_client() -> QdrantClient:
    if os.environ.get("QDRANT_URL"):
        return QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY"))
    return QdrantClient(":memory:")


# ----------------------------------------------------------------------------- the search stack

class Stack:
    """Documents searched by BM25, by dense vectors in Qdrant, or both fused (hybrid). Filters apply to both halves.
    `vectors` lets a caller supply the chunk vectors itself (late chunking does)."""

    def __init__(self, docs: list[Document], embedder, label: str, vectors: np.ndarray | None = None):
        plain = [Document(id=d.id, content=d.content, meta=dict(d.meta)) for d in docs]
        if vectors is None:
            vectors = embedder.docs([d.content for d in plain])
        embedded = [dataclasses.replace(d, embedding=v.tolist()) for d, v in zip(plain, vectors)]
        self.dense_store = QdrantDocumentStore(
            index=collection_name(label, [d.id for d in plain], embedder.dims), embedding_dim=embedder.dims,
            recreate_index=True, similarity="cosine", return_embedding=False, wait_result_from_api=True,
            progress_bar=False, **qdrant_kwargs())
        self.dense_store.write_documents(embedded)
        self.keyword_store = InMemoryDocumentStore(index=f"{label}-kw-{uuid.uuid4().hex[:8]}")
        self.keyword_store.write_documents(plain)
        self.by_id = {d.meta.get("chunk_id", d.id): d for d in plain}
        self.size = len(plain)

        # dense only: question -> vector -> nearest chunks in Qdrant
        self.dense_pipe = Pipeline()
        self.dense_pipe.add_component("embedder", TextEmbedder(embedder))
        self.dense_pipe.add_component("dense", QdrantEmbeddingRetriever(self.dense_store))
        self.dense_pipe.connect("embedder.embedding", "dense.query_embedding")

        # hybrid: BM25 and dense side by side, merged with reciprocal rank fusion
        self.hybrid_pipe = Pipeline()
        self.hybrid_pipe.add_component("embedder", TextEmbedder(embedder))
        self.hybrid_pipe.add_component("dense", QdrantEmbeddingRetriever(self.dense_store))
        self.hybrid_pipe.add_component("bm25", InMemoryBM25Retriever(self.keyword_store))
        self.hybrid_pipe.add_component("joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion"))
        self.hybrid_pipe.connect("embedder.embedding", "dense.query_embedding")
        self.hybrid_pipe.connect("dense.documents", "joiner.documents")
        self.hybrid_pipe.connect("bm25.documents", "joiner.documents")

    def dense(self, query: str, k: int = 5, filters: dict | None = None) -> list[Document]:
        out = self.dense_pipe.run({"embedder": {"text": query}, "dense": {"top_k": k, "filters": filters}})
        return out["dense"]["documents"]

    def bm25(self, query: str, k: int = 5, filters: dict | None = None) -> list[Document]:
        return InMemoryBM25Retriever(self.keyword_store).run(query=query, top_k=k, filters=filters)["documents"]

    def hybrid(self, query: str, k: int = 5, keyword_query: str | None = None, filters: dict | None = None) -> list[Document]:
        """`keyword_query` lets HyDE feed the pretend answer to the dense side and the real question to BM25."""
        out = self.hybrid_pipe.run({"embedder": {"text": query},
                                    "bm25": {"query": keyword_query or query, "top_k": k, "filters": filters},
                                    "dense": {"top_k": k, "filters": filters}, "joiner": {"top_k": k}})
        return out["joiner"]["documents"]


class MultiVectorIndex:
    """ColBERT-style late interaction. Every chunk is stored as MANY vectors (one per token) in a Qdrant multi-vector
    field. A query is also many vectors, and Qdrant scores each chunk with MaxSim (best match per query token, summed)."""

    def __init__(self, docs: list[Document], embedder, label: str = "colbert"):
        self.embedder, self.docs = embedder, list(docs)
        vectors = embedder.tokens([d.content for d in self.docs])
        dim = vectors[0].shape[1]
        self.client, self.name = qdrant_client(), collection_name(label, [d.id for d in self.docs], dim)
        if self.client.collection_exists(self.name):
            self.client.delete_collection(self.name)
        self.client.create_collection(self.name, vectors_config=models.VectorParams(
            size=dim, distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)))
        self.client.upsert(self.name, points=[models.PointStruct(id=i, vector=v.tolist(), payload={"chunk_id": d.meta["chunk_id"]})
                                              for i, (d, v) in enumerate(zip(self.docs, vectors))])

    def search(self, query: str, k: int = 5) -> list[Document]:
        qv = self.embedder.tokens([query], is_query=True)[0]
        hits = self.client.query_points(self.name, query=qv.tolist(), limit=k).points
        return [dataclasses.replace(self.docs[h.id], score=float(h.score)) for h in hits]


# ----------------------------------------------------------------------------------------- the index

class Index:
    def __init__(self, docs_dir: Path = DOCS_DIR):
        self.docs_dir = Path(docs_dir)
        self.chunks = load_chunks(self.docs_dir)
        self.embedder = get_embedder([c.content for c in self.chunks])
        self.plain = Stack(self.chunks, self.embedder, "plain")
        self.standins: list[str] = []          # what ran as an offline stand-in while building
        if not self.embedder.real:
            self.standins.append(f"embeddings: {self.embedder.status}"
                                 + (f" ({self.embedder.fallback_reason})" if hasattr(self.embedder, "fallback_reason") else ""))

    @cached_property
    def by_id(self) -> dict[str, Document]:
        return {c.meta["chunk_id"]: c for c in self.chunks}

    @cached_property
    def corpus_hash(self) -> str:
        """Changes whenever any chunk's text changes. The semantic cache uses it so edited documents never serve old answers."""
        return hashlib.sha1("\n".join(c.content for c in self.chunks).encode()).hexdigest()[:12]

    @cached_property
    def values(self) -> dict[str, list[str]]:
        """Which metadata values exist in the corpus. The filter step may only choose from these."""
        return {f: sorted({c.meta[f] for c in self.chunks}) for f in ("doc_type", "period")}

    # ---- sentence window ---------------------------------------------------------------------
    @cached_property
    def sentences(self):
        """Search single sentences (a sentence is a sharp match for one idea), then hand back the sentences around the hit,
        because the answer is often in the next sentence. Haystack's SentenceWindowRetriever does the widening."""
        splitter = DocumentSplitter(split_by="function", splitting_function=split_sentences)
        sents = []
        for path in sorted(self.docs_dir.glob("*.md")):
            text = path.read_text()
            docs = splitter.run([Document(content=text, meta={"doc_id": path.stem, **doc_meta(path.stem)})])["documents"]
            for d in docs:
                cid = f"{path.stem}#s{d.meta['split_id']}"
                sents.append(dataclasses.replace(d, id=cid, meta={**d.meta, "chunk_id": cid}))
        store = InMemoryDocumentStore(index=f"sentences-{uuid.uuid4().hex[:8]}")
        store.write_documents(sents)
        return Stack(sents, self.embedder, "sentences"), SentenceWindowRetriever(store, window_size=2), sents

    # ---- late chunking -----------------------------------------------------------------------
    @cached_property
    def late(self) -> Stack:
        """Late chunking: run each WHOLE document through a long-context model once, then cut the token vectors into chunk
        vectors. Every chunk vector has 'read' the rest of its document. With no long-context model available this uses a
        toy encoder that mixes each word with its neighbours across the document, so the pooling code still runs for real."""
        encoder = None
        if self.embedder.real and os.environ.get("EMBEDDER", "auto") != "lsa":
            try:
                encoder = HFLateChunker()
            except Exception:
                encoder = None
        if encoder is None:
            base = self.embedder if isinstance(self.embedder, LsaEmbedder) else LsaEmbedder([c.content for c in self.chunks])
            encoder = ToyContextEncoder(base)
            self.standins.append("late chunking: toy moving-average context, not a long-context model")
        vectors, ordered = [], []
        for doc_id in sorted({c.meta["doc_id"] for c in self.chunks}):
            mine = [c for c in self.chunks if c.meta["doc_id"] == doc_id]
            vecs = encoder.encode_doc(full_text(doc_id, self.docs_dir), [(c.meta["char_start"], c.meta["char_end"]) for c in mine])
            vectors += list(vecs)
            ordered += mine
        return Stack(ordered, encoder, "late", vectors=np.array(vectors))

    # ---- ColBERT -----------------------------------------------------------------------------
    @cached_property
    def colbert(self) -> MultiVectorIndex:
        if not self.embedder.real:
            self.standins.append("colbert: each word's vector comes from the TF-IDF+SVD model, not a ColBERT model")
        return MultiVectorIndex(self.chunks, self.embedder)

    # ---- contextual retrieval --------------------------------------------------------------
    @cached_property
    def contextual(self) -> Stack:
        """Anthropic's contextual retrieval: before indexing, a model writes one sentence per chunk placing it
        in its document ('This is the results section of the Leaf Local campaign report...'). The sentence is
        stored WITH the chunk, so a chunk that only says 'Spend was 6.2 million' can still be found by
        'how much did Leaf Local cost?'"""
        out = []
        for c in self.chunks:
            ctx = cached("contextual", f"v1|{mode()}|{c.id}|{c.content}", lambda c=c: self._context_line(c))
            d = Document(id=c.id, content=f"{ctx}\n{c.content}", meta={**c.meta, "context": ctx})
            out.append(d)
        if not have_key():
            self.standins.append("contextual: used 'document title + section' as the context line (no Claude key)")
        return Stack(out, self.embedder, "contextual")

    def _context_line(self, c: Document) -> str:
        whole = full_text(c.meta["doc_id"], self.docs_dir)
        prompt = (f"<document>\n{whole}\n</document>\nHere is a chunk from that document:\n<chunk>\n{c.content}\n</chunk>\n"
                  "Write ONE short sentence that places this chunk within the document, naming the campaign, "
                  "audience or topic it is about, to help search find it. Reply with only that sentence.")
        offline = f"From '{c.meta['title']}', section {c.meta['section']}."
        return ask("You write search context for document chunks.", prompt, "contextualize", max_tokens=80, offline=offline)

    # ---- parent-document ---------------------------------------------------------------------
    @cached_property
    def parents(self):
        """Search small pieces (precise), return the bigger piece around them (enough context to answer).
        If most of a parent's small pieces match, AutoMergingRetriever returns the parent instead of the pieces."""
        splitter = HierarchicalDocumentSplitter(block_sizes={120, 40}, split_by="word")
        leaves, parents = [], []
        for path in sorted(self.docs_dir.glob("*.md")):
            text = path.read_text()
            title = re.search(r"^# (.+)$", text, re.M).group(1).strip()
            docs = splitter.run([Document(content=text, meta={"doc_id": path.stem, "title": title})])["documents"]
            by_level = {}
            for d in docs:
                by_level.setdefault(d.meta["__level"], []).append(d)
            for level, items in by_level.items():
                for i, d in enumerate(items):
                    cid = path.stem if level == 0 else f"{path.stem}/part{level}.{i + 1}"
                    d.meta["chunk_id"] = cid
                    (leaves if level == max(by_level) else parents).append(d)
        parent_store = InMemoryDocumentStore(index=f"parents-{uuid.uuid4().hex[:8]}")
        parent_store.write_documents(parents + leaves)       # the merger looks parents up by id
        return Stack(leaves, self.embedder, "leaves"), AutoMergingRetriever(parent_store, threshold=0.4)

    # ---- RAPTOR -------------------------------------------------------------------------------
    @cached_property
    def raptor(self) -> Stack:
        """RAPTOR: cluster the chunks, summarise each cluster, then cluster the summaries and summarise those,
        until one summary is left. All levels go into one index ('collapsed tree'), so a broad question can match
        a high-level summary and a narrow one can match a single chunk. Clustering is KMeans on the LSA vectors.
        The paper uses UMAP + GMM; KMeans is the simple version of the same idea."""
        nodes = [Document(id=c.id, content=c.content, meta={**c.meta, "level": 0, "members": [c.meta["chunk_id"]]})
                 for c in self.chunks]
        level, current = 0, nodes
        while len(current) > 3 and level < 3:
            vectors = self.embedder.docs([d.content for d in current])
            k = max(2, round(len(current) / 5))
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(vectors)
            nxt = []
            for g in range(k):
                members = [d for d, lab in zip(current, labels) if lab == g]
                if not members:
                    continue
                text = "\n\n".join(m.content for m in members)
                summary = cached("raptor", f"v1|{mode()}|{level}|{text}", lambda text=text: self._summarise(text))
                ids = sorted({i for m in members for i in m.meta["members"]})
                cid = f"raptor/L{level + 1}-{len(nxt) + 1}"
                nxt.append(Document(id=cid, content=summary,
                                    meta={"chunk_id": cid, "level": level + 1, "members": ids,
                                          "title": "Summary of " + ", ".join(sorted({i.split('/')[0] for i in ids})[:3]),
                                          "section": f"level {level + 1}"}))
            nodes += nxt
            current, level = nxt, level + 1
        if not have_key():
            self.standins.append("raptor: summaries are the first sentence of each chunk, not a model summary (no Claude key)")
        return Stack(nodes, self.embedder, "raptor")

    def _summarise(self, text: str) -> str:
        offline = " ".join(re.split(r"(?<=[.!?])\s", t.split("\n", 1)[-1].strip())[0] for t in text.split("\n\n"))[:700]
        return ask("You summarise marketing report passages. Keep every name and number that matters.",
                   f"Summarise these passages in 4 to 5 sentences:\n\n{text[:6000]}", "summarise",
                   model=STRONG, max_tokens=300, offline=offline)

    # ---- GraphRAG (small version) ---------------------------------------------------------
    @cached_property
    def graph(self) -> "Graph":
        if not have_key():
            self.standins.append("graph: entities found by capitalisation and repeated word pairs, not by a model (no Claude key)")
        g = Graph(self.chunks, self.docs_dir)
        if g.store.kind == "memory":
            self.standins.append("graph store: in memory with pure-Python PageRank and grouping (set GRAPH_BACKEND=neo4j for Neo4j + GDS)")
        return g


# ---------------------------------------------------------------------------------------- the graph

CAP_PHRASE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[ ](?:&[ ])?(?:[A-Z][A-Za-z0-9]*|Co))+\b")   # spaces only, never across a line
CAMEL = re.compile(r"\b[A-Z][a-z]+[A-Z][A-Za-z]*\b")                                       # TikTok, YouTube


def rule_entities(text: str, lowercase_words: set[str]) -> list[str]:
    """Offline entity finder: capitalised phrases ('Northbeam Media', 'Ledger & Co') and CamelCase words (TikTok).
    A capitalised phrase at the start of a sentence often begins with an ordinary word ('Planned Crunch Time').
    If that first word also appears in lowercase somewhere in the documents, it is an ordinary word, so I drop it
    (as long as at least two words are left, so 'Value Shoppers' survives)."""
    out = []
    for m in CAP_PHRASE.finditer(text):
        words = m.group(0).split(" ")
        if words[0].lower() in lowercase_words and len(words) > 2:
            words = words[1:]
        elif words[0].lower() in lowercase_words and len(words) == 2 and words[1] not in ("&",):
            at_start = m.start() == 0 or text[:m.start()].rstrip().endswith((".", "\n"))
            if at_start and words[0] in ("A", "The", "For", "From", "This", "That", "After", "Early"):
                continue
        out.append(" ".join(words))
    return out + CAMEL.findall(text)


def repeated_pairs(chunks: list[Document]) -> set[str]:
    """Word pairs that show up in several chunks ('retail media', 'paid search'). Mechanical, no list typed by hand."""
    seen: dict[str, set[str]] = {}
    for c in chunks:
        words = [w for w in re.findall(r"[a-z]+", c.content.lower())]
        for a, b in zip(words, words[1:]):
            if len(a) > 3 and len(b) > 3 and a not in STOP and b not in STOP:
                seen.setdefault(f"{a} {b}", set()).add(c.meta["chunk_id"])
    return {p for p, ids in seen.items() if len(ids) >= 3}


STOP = {"that", "this", "with", "from", "have", "were", "than", "their", "which", "about", "more", "most", "into",
        "also", "such", "because", "after", "when", "they", "them", "there", "will", "would", "should", "could",
        "dollars", "million", "percent", "spend", "every", "other", "first", "each", "only", "even", "while"}


class Graph:
    """Entities as dots, 'appear in the same chunk' as lines. Two kinds of question use it:
      local   'Which agency ran the campaign that returned 2.6?'  start from the entities in the question,
              spread outwards with personalised PageRank, then return the chunks that mention the top entities.
      global  'What are the main lessons across campaigns?'  find communities of entities and search a short
              report for each community.
    The graph itself lives in a store (graph_store.py): in memory by default, or in Neo4j with GDS running the algorithms."""

    def __init__(self, chunks: list[Document], docs_dir: Path, store=None):
        self.chunks = chunks
        self.store = store or get_store()
        pairs = repeated_pairs(chunks)
        self.lower = {w for c in chunks for w in re.findall(r"\b[a-z]+\b", c.content)}      # every word seen in lowercase
        chunk_entities: dict[str, set[str]] = {}
        for c in chunks:
            ents = cached("entities", f"v1|{mode()}|{c.id}|{c.content}", lambda c=c: self._entities(c, pairs))
            chunk_entities[c.meta["chunk_id"]] = set(ents)
        counts: dict[str, int] = {}
        for ents in chunk_entities.values():
            for e in ents:
                counts[e] = counts.get(e, 0) + 1
        too_common = max(4, int(0.25 * len(chunks)))              # an entity in a quarter of all chunks links everything
        keep = {e for e, n in counts.items() if n <= too_common}
        mentions: dict[str, list[str]] = {}
        edges: dict[tuple[str, str], float] = {}
        for cid, ents in chunk_entities.items():
            ents = sorted(ents & keep)
            for e in ents:
                mentions.setdefault(e, []).append(cid)
            for i, a in enumerate(ents):
                for b in ents[i + 1:]:
                    edges[(a, b)] = edges.get((a, b), 0) + 1
        self.store.load(mentions, edges)
        self.by_id = {c.meta["chunk_id"]: c for c in chunks}
        self._names = self.store.entity_names()
        self.reports = self._reports()
        self.report_search = None

    def _entities(self, c: Document, pairs: set[str]) -> list[str]:
        live = _json_list(ask("You list the named things in a text: companies, agencies, brands, products, "
                              "campaigns, audiences, channels. Reply with ONLY a JSON list of short strings.",
                              c.content, "entities", max_tokens=200, offline=""))
        names = [x for x in live if isinstance(x, str)]
        if not names:
            low = c.content.lower()
            names = rule_entities(c.content, self.lower) + [p for p in pairs if p in low]
        return sorted({n.strip().lower() for n in names if len(n.strip()) > 2})

    # ---- local ----------------------------------------------------------------------------
    def seeds(self, question: str) -> list[str]:
        """Entities named in the question. Longest names first, and each match is blanked out so that
        'Leaf Crisps' is not also counted as 'Leaf'."""
        q = " " + question.lower() + " "
        found = []
        for name in sorted(self._names, key=len, reverse=True):
            if f" {name.lower()} " in q.replace("?", " ").replace(",", " ").replace(".", " ").replace("'s", " "):
                found.append(name)
                q = q.replace(name.lower(), " " * len(name))
        return found

    def local(self, question: str, k: int = 5, top_entities: int = 6):
        seeds = self.seeds(question)
        if not seeds:
            return [], seeds, []
        scores = self.store.pagerank(seeds)                       # in memory, or GDS personalised PageRank inside Neo4j
        top = sorted(scores, key=lambda n: (-scores[n], n))[:top_entities]
        chunk_score: dict[str, float] = {}
        for entity, chunk_ids in self.store.mentions_of(top).items():
            for cid in chunk_ids:
                chunk_score[cid] = chunk_score.get(cid, 0) + scores[entity]
        ranked = sorted(chunk_score, key=lambda c: (-chunk_score[c], c))[:k]
        return [self.by_id[c] for c in ranked], seeds, top

    # ---- global ---------------------------------------------------------------------------
    def _reports(self) -> list[Document]:
        groups = self.store.communities()                         # in memory, or GDS Leiden inside Neo4j
        mentions = self.store.mentions_of(self._names)
        reports = []
        for group in sorted(groups, key=lambda g: (-len(g), g)):
            if len(group) < 2:
                continue
            ids = sorted({cid for e in group for cid in mentions.get(e, ())})
            text = "\n\n".join(self.by_id[c].content for c in ids)[:5000]
            names = sorted(group, key=lambda e: (-len(mentions[e]), e))[:6]
            offline = " ".join(self.by_id[c].content.split("\n", 1)[-1].split(". ")[0].strip() + "." for c in ids[:4])
            summary = cached("reports", f"v1|{mode()}|{','.join(names)}|{text}",
                             lambda text=text, names=names, offline=offline: ask(
                                 "You write short theme reports about a group of related marketing entities.",
                                 f"Entities: {', '.join(names)}\nWrite 3 to 4 sentences on what these passages say about them.\n\n{text}",
                                 "community_report", model=STRONG, max_tokens=250, offline=offline))
            cid = f"graph/community-{len(reports) + 1}"
            reports.append(Document(id=cid, content=f"Theme: {', '.join(names)}\n{summary}",
                                    meta={"chunk_id": cid, "title": "Community report", "section": ", ".join(names[:3]),
                                          "members": ids, "entities": sorted(group)}))
        self.store.save_reports([{"id": r.id, "members": r.meta["members"], "text": r.content} for r in reports])
        return reports

    def global_(self, question: str, k: int = 3) -> list[Document]:
        if not self.reports:
            return []
        if self.report_search is None:
            store = InMemoryDocumentStore(index=f"reports-{uuid.uuid4().hex[:8]}")
            store.write_documents(self.reports)
            self.report_search = InMemoryBM25Retriever(store)
        return self.report_search.run(query=question, top_k=k)["documents"]
