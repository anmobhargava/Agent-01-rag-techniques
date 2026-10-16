"""
Semantic cache: if someone asks something that means the same as a question already answered, reuse that answer.
It costs nothing, takes milliseconds, and skips every model call.

How it works: each answered question is stored in a Qdrant collection as a vector. A new question is turned into a vector,
and if its nearest stored question is close enough (cosine >= threshold) the stored answer comes back.

The trap: similarity alone is not safe here. To a vector model, "How much did Crunch Time spend?" and "How much did
Leaf Local spend?" are nearly the same sentence, and their answers are different. A cache that returns the wrong campaign's
number is worse than no cache. So a hit needs THREE things:
  1. similarity above the threshold,
  2. the same "key facts" in both questions: the same names (Crunch Time), numbers (2.6) and periods (Q3),
  3. the same corpus (a hash of the chunks) and the same mode (live Claude vs offline), and the entry is not older than the TTL.

What gets cached: answers that were produced and passed the citation check. Never refusals, blocked questions or answers
with unknown citations. The threshold is higher for the TF-IDF stand-in (0.97) than for a neural model (0.92), because
the stand-in puts unrelated questions closer together. Tune the threshold on your own questions before trusting it.
"""

import dataclasses
import os
import re
import time
import uuid

from qdrant_client import models

from .index import collection_name, qdrant_client

CACHEABLE = ("ok", "no_api_key_extractive")


def key_facts(question: str) -> list[str]:
    """Names, numbers and periods in the question, lowercased. Two questions can only share a cache entry if these match."""
    named = re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)+|[A-Z][a-z]+[A-Z]\w*", question)       # 'Crunch Time', 'TikTok'
    numbers = re.findall(r"(?<![A-Za-z])\d[\d.,%]*", question)      # not the 3 in Q3
    periods = re.findall(r"\b[hq][1-4]\b", question.lower())
    return sorted({x.lower().strip(".,") for x in named + numbers + periods})


class SemanticCache:
    def __init__(self, embedder, corpus_hash: str, threshold: float | None = None, ttl: float = 3600.0, max_size: int = 256):
        self.embedder, self.corpus_hash, self.ttl, self.max_size = embedder, corpus_hash, ttl, max_size
        default = 0.92 if embedder.real else 0.97
        self.threshold = float(os.environ.get("CACHE_THRESHOLD", threshold if threshold is not None else default))
        self.client = qdrant_client()
        self.name = collection_name("cache", [corpus_hash], embedder.dims)
        if self.client.collection_exists(self.name):
            self.client.delete_collection(self.name)
        self.client.create_collection(self.name, vectors_config=models.VectorParams(size=embedder.dims, distance=models.Distance.COSINE))
        self.order: list[str] = []                     # oldest first, so the cache can evict
        self.hits = self.misses = 0

    def get(self, question: str, live: bool):
        """Return (cached_answer_dict, similarity) or None."""
        vec = self.embedder.query(question).tolist()
        found = self.client.query_points(self.name, query=vec, limit=3, with_payload=True).points
        for h in found:
            p = h.payload
            if (h.score >= self.threshold and p["corpus_hash"] == self.corpus_hash and p["live"] == live
                    and p["facts"] == key_facts(question) and time.time() - p["ts"] <= self.ttl):
                self.hits += 1
                return p["answer"], float(h.score)
        self.misses += 1
        return None

    def put(self, question: str, answer, live: bool) -> bool:
        """Store an answer. Returns False (and stores nothing) if this kind of answer must not be cached."""
        if answer.status not in CACHEABLE or answer.unknown_citations:
            return False
        pid = str(uuid.uuid4())
        payload = {"answer": dataclasses.asdict(answer), "facts": key_facts(question), "live": live,
                   "corpus_hash": self.corpus_hash, "ts": time.time(), "question": question}
        self.client.upsert(self.name, points=[models.PointStruct(id=pid, vector=self.embedder.query(question).tolist(), payload=payload)])
        self.order.append(pid)
        while len(self.order) > self.max_size:
            self.client.delete(self.name, points_selector=models.PointIdsList(points=[self.order.pop(0)]))
        return True

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self.order), "threshold": self.threshold}
