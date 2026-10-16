"""
The retrieval techniques. Each one takes a question and returns a Retrieval: the chunks it found, plus
a few plain-words notes on what it did (the app shows those notes).

How to read this file: every technique is one short function. Most of them are a Haystack pipeline
(built in index.py or components.py) with a few lines around it.

  naive         dense vectors only. The baseline everything else is compared to.
  hybrid        BM25 keywords + dense vectors, fused with reciprocal rank fusion.
  rerank        hybrid for 15 candidates, then reorder them by reading question and chunk together.
  hyde          write a pretend answer, search with that instead of the question.
  multi_query   ask the question several ways, fuse the results.
  decompose     split a two-part question into smaller ones, search each.
  parent_doc    search small pieces, return the bigger piece around them.
  contextual    every chunk carries a sentence saying where it came from.
  raptor        search summaries of summaries (a tree), so broad questions find broad answers.
  graphrag      entities and links. Local for "how is X connected to Y", global for "what are the themes".
  corrective    retrieve, grade, rewrite and retry when the chunks are poor, drop the junk.
  sentence_window  search single sentences, return the sentences around the hit.
  late_chunking    embed the whole document first, then cut it into chunk vectors.
  colbert       one vector per word, scored with MaxSim (late interaction), stored in Qdrant.
  filtered      metadata filtering: only search documents of the kind or period the question names.
  iterative     adaptive multi-step: search, see what is still missing, search again.
"""

from dataclasses import dataclass, field
from typing import Callable

import dataclasses

from haystack import Document, Pipeline
from haystack.components.joiners import DocumentJoiner
from haystack.components.routers import ConditionalRouter

from .components import (ClaudeReranker, FilterExtractor, GapFinder, HydeQuery, OverlapReranker, QueryExpander,
                         RelevanceGrader, terms)
from .index import Index
from .llm import ask, have_key

GOOD_ENOUGH = 0.2          # a retrieval is "good enough" when at least one in five chunks is judged relevant


@dataclass
class Retrieval:
    technique: str
    documents: list[Document]
    notes: list[str] = field(default_factory=list)       # what happened, in plain words
    standins: list[str] = field(default_factory=list)    # which parts were offline stand-ins, not the real thing
    graded: float | None = None                          # share of chunks judged relevant, when a grader ran


def fuse(lists: list[list[Document]], k: int) -> list[Document]:
    """Merge several ranked lists into one with reciprocal rank fusion (Haystack's joiner does the maths)."""
    return DocumentJoiner(join_mode="reciprocal_rank_fusion").run(documents=lists, top_k=k)["documents"]


# ------------------------------------------------------------------------------------------ the basics

def naive(ix: Index, q: str, k: int) -> Retrieval:
    return Retrieval("naive", ix.plain.dense(q, k), ["Dense vectors only, no keywords, no reranking."],
                     ["dense vectors are TF-IDF+SVD, not a neural embedding"])


def hybrid(ix: Index, q: str, k: int) -> Retrieval:
    return Retrieval("hybrid", ix.plain.hybrid(q, k),
                     ["Ran BM25 and dense search side by side and fused the two rankings (reciprocal rank fusion)."],
                     ["dense vectors are TF-IDF+SVD, not a neural embedding"])


def rerank(ix: Index, q: str, k: int) -> Retrieval:
    candidates = ix.plain.hybrid(q, 15)
    reranker = ClaudeReranker() if have_key() else OverlapReranker()
    best = reranker.run(candidates, q, k)["documents"]
    kind = "Claude read all 15 candidates against the question" if have_key() else "an overlap score (no Claude key)"
    return Retrieval("rerank", best, [f"Hybrid search found 15 candidates, then {kind} picked the top {k}."],
                     [] if have_key() else ["reranker is a word-overlap score, not a cross-encoder or Claude"])


# ------------------------------------------------------------------------ changing the query

def hyde(ix: Index, q: str, k: int) -> Retrieval:
    out = HydeQuery().run(q)
    if out["was_standin"]:
        docs = ix.plain.hybrid(q, k)
        return Retrieval("hyde", docs, ["No model available, so HyDE could not write a pretend answer and searched with the plain question."],
                         ["HyDE needs a model to write the pretend answer; this run was plain hybrid search"])
    # the pretend answer goes to the dense side (answers look like answers); the real question still feeds BM25
    docs = ix.plain.hybrid(out["text"], k, keyword_query=q)
    return Retrieval("hyde", docs, [f"Claude wrote a pretend answer: \"{out['text'][:200]}\". I searched with that, not the question."])


def multi_query(ix: Index, q: str, k: int) -> Retrieval:
    out = QueryExpander("paraphrase").run(q)
    lists = [ix.plain.hybrid(x, k) for x in out["queries"]]
    note = f"Searched {len(lists)} versions of the question and fused the results: " + " | ".join(out["queries"])
    return Retrieval("multi_query", fuse(lists, k), [note],
                     ["paraphrases came from a rule (key words only), not Claude"] if out["was_standin"] else [])


def decompose(ix: Index, q: str, k: int) -> Retrieval:
    out = QueryExpander("decompose").run(q)
    lists = [ix.plain.hybrid(x, max(3, k // 2 + 1)) for x in out["queries"]]
    note = "Split into smaller questions: " + " | ".join(out["queries"])
    return Retrieval("decompose", fuse(lists, k), [note],
                     ["split is a rule (on 'and'), so later parts may lose context; Claude does this properly"] if out["was_standin"] else [])


# ------------------------------------------------------------------------ changing what is indexed

def parent_doc(ix: Index, q: str, k: int) -> Retrieval:
    leaves, merger = ix.parents
    hits = leaves.hybrid(q, k + 3)
    merged = merger.run(hits)["documents"]
    bigger = sum(1 for d in merged if d.meta.get("__level") != hits[0].meta.get("__level")) if hits else 0
    return Retrieval("parent_doc", merged[:k],
                     [f"Searched {len(hits)} small pieces; returned {len(merged)} items, {bigger} of them a bigger parent piece "
                      "because most of its small pieces matched."])


def contextual(ix: Index, q: str, k: int) -> Retrieval:
    docs = ix.contextual.hybrid(q, k)
    first = docs[0].meta.get("context", "") if docs else ""
    standin = [] if have_key() else ["the context line is 'document title + section', not written by Claude"]
    return Retrieval("contextual", docs, [f"Every chunk was indexed with a context line. Top hit's line: \"{first}\""], standin)


def raptor(ix: Index, q: str, k: int) -> Retrieval:
    docs = ix.raptor.hybrid(q, k)
    levels = sorted({d.meta.get("level", 0) for d in docs})
    return Retrieval("raptor", docs, [f"Searched the whole summary tree at once. Matches came from level(s) {levels} "
                                      "(0 = an original chunk, 1+ = a summary of summaries)."],
                     [] if have_key() else ["tree summaries are the first sentence of each chunk, not model summaries"])


def graphrag(ix: Index, q: str, k: int) -> Retrieval:
    g = ix.graph
    standin = [] if have_key() else ["entities were found by capitalisation and repeated word pairs, not by Claude"]
    docs, seeds, top = g.local(q, k)
    if docs:
        return Retrieval("graphrag", docs, [f"Local path. Found {seeds} in the question, walked the entity graph "
                                            f"(personalised PageRank), top entities: {top}. Returned chunks mentioning them."], standin)
    docs = g.global_(q, min(k, 3))
    return Retrieval("graphrag", docs, ["Global path. No known entity in the question, so I searched community reports "
                                        "(short summaries of groups of related entities)."], standin)


# ----------------------------------------------------------------------------- more ways to index and search

def sentence_window(ix: Index, q: str, k: int) -> Retrieval:
    stack, window_retriever, sents = ix.sentences
    hits = stack.hybrid(q, k)
    windows = window_retriever.run(retrieved_documents=hits)["context_windows"]
    docs = []
    for hit, text in zip(hits, windows):
        near = [s for s in sents if s.meta["doc_id"] == hit.meta["doc_id"] and abs(s.meta["split_id"] - hit.meta["split_id"]) <= 2]
        start = min(s.meta["split_idx_start"] for s in near)
        end = max(s.meta["split_idx_start"] + len(s.content) for s in near)
        # which section chunks does this window touch? (so scoring and citations can point at real chunks)
        members = [c.meta["chunk_id"] for c in ix.chunks
                   if c.meta["doc_id"] == hit.meta["doc_id"] and c.meta["char_start"] < end and c.meta["char_end"] > start]
        cid = hit.meta["chunk_id"] + "+window"
        docs.append(dataclasses.replace(hit, id=cid, content=text.strip(), meta={**hit.meta, "chunk_id": cid, "members": members,
                                                                                  "title": hit.meta.get("title", hit.meta["doc_id"]), "section": "sentence window"}))
    return Retrieval("sentence_window", docs, [f"Searched {stack.size} single sentences, then widened each of the top {len(hits)} hits to the sentence "
                                                "before and after it (window of 2 each side)."])


def late_chunking(ix: Index, q: str, k: int) -> Retrieval:
    docs = ix.late.hybrid(q, k)
    toy = any("late chunking" in s for s in ix.standins)               # the index records when it fell back to the toy encoder
    return Retrieval("late_chunking", docs, ["Each chunk vector was made by embedding the WHOLE document first, then averaging that chunk's tokens, "
                                             "so it carries context from the rest of its document."],
                     ["late chunking used a toy moving-average encoder, not a long-context model"] if toy else [])



def colbert(ix: Index, q: str, k: int) -> Retrieval:
    docs = ix.colbert.search(q, k)
    return Retrieval("colbert", docs, ["Late interaction: every chunk is stored as one vector per word. Each query word finds its best match in the "
                                       "chunk (MaxSim), and the best matches are added up. Qdrant does the scoring."],
                     [] if ix.embedder.real else ["word vectors come from the TF-IDF+SVD model, not a ColBERT model"])


def filtered(ix: Index, q: str, k: int) -> Retrieval:
    f = FilterExtractor(ix.values).run(q)
    standin = ["filter chosen by matching words in the question, not by Claude"] if f["was_standin"] else []
    if f["filters"] is None:
        return Retrieval("filtered", ix.plain.hybrid(q, k), ["The question names no document type or period, so no filter was applied (plain hybrid)."], standin)
    docs = ix.plain.hybrid(q, k, filters=f["filters"])
    if not docs:
        return Retrieval("filtered", ix.plain.hybrid(q, k), [f"Filter ({f['why']}) matched nothing, so I searched everything."], standin)
    return Retrieval("filtered", docs, [f"Applied a metadata filter ({f['why']}) to BOTH the keyword and the vector search before ranking."], standin)


def iterative(ix: Index, q: str, k: int, max_hops: int = 3) -> Retrieval:
    lists, notes, seen = [ix.plain.hybrid(q, k)], [], []
    standin = []
    notes.append(f"Hop 1: searched the question as asked ({len(lists[0])} chunks).")
    for hop in range(2, max_hops + 1):
        so_far = [d for lst in lists for d in lst]
        gap = GapFinder().run(q, so_far)
        if gap["was_standin"] and not standin:
            standin = ["the 'what is missing?' step is a rule (question words no chunk contains), not Claude"]
        if not gap["query"] or gap["query"] in seen:
            notes.append(f"Hop {hop}: nothing missing, stopped.")
            break
        seen.append(gap["query"])
        lists.append(ix.plain.hybrid(gap["query"], k))
        notes.append(f"Hop {hop}: still missing -> searched \"{gap['query']}\" ({len(lists[-1])} chunks).")
    return Retrieval("iterative", fuse(lists, k), notes, standin)


# ----------------------------------------------------------------------------------- self-checking

def _corrective_pipeline() -> Pipeline:
    """grade the chunks, then a router sends the result down 'good' or 'retry'. Plain Haystack, no agent loop."""
    router = ConditionalRouter(routes=[
        {"condition": "{{ score >= %s }}" % GOOD_ENOUGH, "output": "{{ relevant }}", "output_name": "good",
         "output_type": list[Document]},
        {"condition": "{{ score < %s }}" % GOOD_ENOUGH, "output": "{{ query }}", "output_name": "retry",
         "output_type": str},
    ])
    p = Pipeline()
    p.add_component("grader", RelevanceGrader())
    p.add_component("router", router)
    p.connect("grader.score", "router.score")
    p.connect("grader.relevant", "router.relevant")
    return p


_PIPE = _corrective_pipeline()


def grade(q: str, docs: list[Document]) -> dict:
    """Run the grader+router pipeline once. Returns the chunks judged relevant, the score, and the route taken."""
    out = _PIPE.run({"grader": {"query": q, "documents": docs}, "router": {"query": q}},
                    include_outputs_from={"grader"})
    g = out["grader"]
    return {"relevant": g["relevant"], "score": g["score"], "standin": g["was_standin"],
            "route": "good" if "good" in out["router"] else "retry"}


def rewrite(q: str) -> str:
    live = ask("Rewrite the question as a short keyword-style search query for an internal marketing report archive. "
               "Reply with only the query.", q, "rewrite", max_tokens=60, offline="")
    return live or " ".join(terms(q))


def corrective(ix: Index, q: str, k: int) -> Retrieval:
    first = ix.plain.hybrid(q, k + 3)
    g1 = grade(q, first)
    notes = [f"Retrieved {len(first)} chunks; the grader kept {len(g1['relevant'])} (score {g1['score']})."]
    standins = ["the grader is a word-overlap rule, not Claude"] if g1["standin"] else []
    if g1["route"] == "good":
        return Retrieval("corrective", g1["relevant"][:k], notes + ["Good enough, dropped the rest."], standins, g1["score"])
    q2 = rewrite(q)
    second = ix.plain.hybrid(q2, k + 3)
    g2 = grade(q, second)
    notes.append(f"Too few relevant. Rewrote the search to \"{q2}\" and tried again: kept {len(g2['relevant'])} (score {g2['score']}).")
    best = g2 if g2["score"] >= g1["score"] else g1
    if best["route"] == "retry" and not best["relevant"]:
        notes.append("Still nothing relevant. Returning nothing, so the answer step can say it does not know.")
    return Retrieval("corrective", best["relevant"][:k], notes, standins, best["score"])


# ---------------------------------------------------------------------------------------- the menu

@dataclass
class Technique:
    key: str
    label: str
    when: str                      # one line; the router shows this menu to Claude and the app shows it to people
    fn: Callable[[Index, str, int], Retrieval]


TECHNIQUES: dict[str, Technique] = {t.key: t for t in [
    Technique("naive", "Naive (dense only)", "The baseline: vectors only. Use it to see what the others improve.", naive),
    Technique("hybrid", "Hybrid (BM25 + vectors)", "Specific questions with names or numbers. A strong default.", hybrid),
    Technique("rerank", "Hybrid + rerank", "When the right chunk is found but ranked too low.", rerank),
    Technique("hyde", "HyDE", "Vague or abstract questions whose words do not match the documents.", hyde),
    Technique("multi_query", "Multi-query", "Questions where the wording could vary (cost / spend / budget).", multi_query),
    Technique("decompose", "Decomposition", "Questions with two parts, joined by 'and' or 'compared to'.", decompose),
    Technique("parent_doc", "Parent document", "When the answer needs the text around the matching sentence.", parent_doc),
    Technique("contextual", "Contextual retrieval", "Chunks that never name their subject (a 'Results' section).", contextual),
    Technique("raptor", "RAPTOR (summary tree)", "Broad questions: themes, lessons, overall performance.", raptor),
    Technique("graphrag", "GraphRAG", "Questions about how things connect, or what holds across everything.", graphrag),
    Technique("corrective", "Corrective RAG", "When a wrong chunk is worse than no chunk: grade, rewrite, retry.", corrective),
    Technique("sentence_window", "Sentence window", "The answer sits next to the matching sentence: search sentences, return their neighbours.", sentence_window),
    Technique("late_chunking", "Late chunking", "Chunks that lose their meaning when cut out of the document ('the campaign', 'it').", late_chunking),
    Technique("colbert", "ColBERT (late interaction)", "Short, exact queries where every word should match something: names and numbers.", colbert),
    Technique("filtered", "Metadata filter", "The question names a kind of document or a period: 'the research', 'in Q3', 'the campaign reports'.", filtered),
    Technique("iterative", "Iterative (multi-step)", "Questions one search cannot finish: search, see what is missing, search again.", iterative),
]}
