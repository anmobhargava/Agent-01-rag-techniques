"""
The small Haystack components that the techniques are built from.

I use Haystack for what it is good at: wiring steps into a pipeline you can read, and its ready-made
pieces (stores, retrievers, the joiner, the splitter). The technique logic itself (HyDE, query
expansion, grading, reranking) is plain Python inside my own components, so it is easy to read and change.

Each component below does one job:
  TextEmbedder / DocumentEmbedder         turn text into vectors, using whichever Embedder embedders.py chose
  OverlapReranker / ClaudeReranker        put the best chunks first
  HydeQuery                               HyDE: write a pretend answer and search with that
  QueryExpander                           multi-query (paraphrases) and decomposition (sub-questions)
  RelevanceGrader                         corrective RAG: are these chunks actually relevant?
  FilterExtractor                         metadata filtering: turn "in Q3" or "from the research" into a filter
  GapFinder                               iterative retrieval: after reading what we have, what is still missing?
"""

import dataclasses
import json
import os
import re

os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")      # no usage pings from this demo
from haystack import Document, component
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .llm import FAST, ask

# Words that make a question a question but carry no topic. Dropped before I count overlap.
QUESTION_WORDS = {"what", "which", "who", "how", "much", "many", "did", "does", "was", "were", "is", "are",
                  "tell", "about", "me", "give", "show", "us", "any", "all", "ran", "run", "say", "said", "says", "do", "go", "went",
                  "get", "got"}


def terms(text: str) -> list[str]:
    """Lower-cased content words with a very light stem (drop a trailing 's'). Used for overlap checks."""
    words = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    out = []
    for w in words:
        if w in ENGLISH_STOP_WORDS or w in QUESTION_WORDS or len(w) < 2:
            continue
        out.append(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


def coverage(question: str, text: str) -> float:
    """Share of the question's content words that appear in `text`. 0 to 1."""
    q = set(terms(question))
    return len(q & set(terms(text))) / len(q) if q else 0.0


# ------------------------------------------------------------------------------------------- embeddings

@component
class TextEmbedder:
    """Question text -> one vector. Works with any Embedder from embedders.py (neural model, fine-tuned model, stand-in)."""

    def __init__(self, embedder):
        self.embedder = embedder

    @component.output_types(embedding=list[float])
    def run(self, text: str):
        return {"embedding": self.embedder.query(text).tolist()}


@component
class DocumentEmbedder:
    """Documents -> the same documents with a vector attached."""

    def __init__(self, embedder):
        self.embedder = embedder

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        vectors = self.embedder.docs([d.content for d in documents])
        return {"documents": [dataclasses.replace(d, embedding=v.tolist()) for d, v in zip(documents, vectors)]}


# ------------------------------------------------------------------------------------------- reranking

@component
class OverlapReranker:
    """Offline reranker: score each chunk by how much of the question it covers. Cheap and rough.
    The point of a reranker is to look at question and chunk TOGETHER, which this does in a crude way."""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], query: str, top_k: int = 5):
        scored = sorted(documents, key=lambda d: (coverage(query, d.content), d.score or 0), reverse=True)
        return {"documents": scored[:top_k]}


@component
class ClaudeReranker:
    """Live reranker: one Claude call reads the question and all candidates and returns the best order.
    Haystack ships an LLMRanker too. I wrote this one so the call shows up in my own cost trace.
    With no key it falls back to OverlapReranker."""

    SYSTEM = ("You rank text chunks by how well they help answer a question. "
              "Reply with ONLY a JSON list of chunk numbers, best first, e.g. [3,0,2].")

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document], query: str, top_k: int = 5):
        listing = "\n".join(f"[{i}] {d.content[:400]}" for i, d in enumerate(documents))
        fallback = OverlapReranker().run(documents, query, top_k)["documents"]
        reply = ask(self.SYSTEM, f"Question: {query}\n\nChunks:\n{listing}", "rerank", max_tokens=100,
                    offline="")
        order = _json_list(reply)
        valid = [i for i in order if isinstance(i, int) and 0 <= i < len(documents)]
        picked = [documents[i] for i in dict.fromkeys(valid)]    # drop repeated numbers, keep Claude's order
        return {"documents": (picked or fallback)[:top_k]}


def _json_list(text: str) -> list:
    m = re.search(r"\[.*?\]", text or "", re.S)
    if not m:
        return []
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, list) else []
    except json.JSONDecodeError:
        return []


# ------------------------------------------------------------------------------------------------- HyDE

@component
class HydeQuery:
    """HyDE (hypothetical document embeddings). A question and its answer rarely look alike, but an answer
    looks like another answer. So I ask the model to WRITE a short pretend answer and search with that.
    The pretend answer can be wrong. It is only used to find real chunks, never shown as fact."""

    SYSTEM = ("Write a short passage (2 to 3 sentences) from an internal marketing report that would answer "
              "the question. Invent plausible details if you need to. It is only used for search.")

    @component.output_types(text=str, was_standin=bool)
    def run(self, query: str):
        pretend = ask(self.SYSTEM, query, "hyde", max_tokens=150, offline="")
        # offline there is no model to write the pretend answer, so HyDE degrades to a plain search
        return {"text": pretend or query, "was_standin": not pretend}


# ------------------------------------------------------------------------------------------ multi-query

@component
class QueryExpander:
    """mode="paraphrase"  -> 3 different ways to ask the same thing (multi-query)
       mode="decompose"   -> the smaller questions that must each be answered first (decomposition)"""

    def __init__(self, mode: str = "paraphrase"):
        assert mode in ("paraphrase", "decompose")
        self.mode = mode

    PARAPHRASE = ("Rewrite the question 3 different ways, using different words a marketing report might use. "
                  "Reply with ONLY a JSON list of 3 strings.")
    DECOMPOSE = ("Split the question into the smallest separate questions that must each be answered to answer "
                 "it. Each one must make sense on its own (repeat names, never say 'it' or 'that'). "
                 "Reply with ONLY a JSON list of 2 to 4 strings.")

    @component.output_types(queries=list[str], was_standin=bool)
    def run(self, query: str):
        system = self.PARAPHRASE if self.mode == "paraphrase" else self.DECOMPOSE
        out = [q for q in _json_list(ask(system, query, f"expand_{self.mode}", max_tokens=250)) if isinstance(q, str)]
        if out:
            return {"queries": [query] + out if self.mode == "paraphrase" else out, "was_standin": False}
        return {"queries": self._offline(query), "was_standin": True}

    def _offline(self, query: str) -> list[str]:
        """Rule-based stand-in. Paraphrase: the question, then just its key words. Decompose: split on 'and'."""
        if self.mode == "paraphrase":
            keys = " ".join(terms(query))
            return [query] + ([keys] if keys and keys != query.lower() else [])
        parts = [p.strip(" ?,") for p in re.split(r"\band\b|;|\bthen\b|\bversus\b|\bcompared (?:to|with)\b", query)]
        parts = [p for p in parts if terms(p)]
        # a fragment like "what is its fee" has lost its subject. Borrow the last name from the part before it.
        names = [m for m in re.findall(r"[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*", parts[0] if parts else "") if m.lower() not in QUESTION_WORDS]
        fixed = [parts[0]] if parts else []
        for p in parts[1:]:
            fixed.append(f"{p} (about {names[-1]})" if names and re.search(r"\b(it|its|that|their|this)\b", p, re.I) else p)
        return [p + "?" for p in fixed] or [query]


# --------------------------------------------------------------------------------------------- grading

@component
class RelevanceGrader:
    """Corrective RAG, step one: look at what was retrieved and say which chunks really help.
    Live: one Claude call. Offline: a chunk counts as relevant when it covers at least a third of the question's
    content words. `score` is the share of chunks judged relevant."""

    SYSTEM = ("You check retrieved text chunks against a question. A chunk is relevant if it contains "
              "information that helps answer the question. Reply with ONLY a JSON list of the relevant chunk "
              "numbers, e.g. [0,2]. Reply [] if none help.")

    @component.output_types(relevant=list[Document], irrelevant=list[Document], score=float, was_standin=bool)
    def run(self, query: str, documents: list[Document]):
        if not documents:
            return {"relevant": [], "irrelevant": [], "score": 0.0, "was_standin": False}
        listing = "\n".join(f"[{i}] {d.content[:400]}" for i, d in enumerate(documents))
        reply = ask(self.SYSTEM, f"Question: {query}\n\nChunks:\n{listing}", "grade", max_tokens=60, offline="NONE")
        standin = reply == "NONE"
        if standin:
            keep = [i for i, d in enumerate(documents) if coverage(query, d.content) >= 0.34]
        else:
            keep = [i for i in _json_list(reply) if isinstance(i, int) and 0 <= i < len(documents)]
        rel = [d for i, d in enumerate(documents) if i in keep]
        irr = [d for i, d in enumerate(documents) if i not in keep]
        return {"relevant": rel, "irrelevant": irr, "score": round(len(rel) / len(documents), 3),
                "was_standin": standin}


# --------------------------------------------------------------------------------------- metadata filtering

DOC_TYPE_WORDS = {"campaign": "campaign", "campaigns": "campaign", "research": "research", "review": "review",
                  "reviews": "review", "scorecard": "scorecard", "playbook": "playbook", "guideline": "guidelines",
                  "guidelines": "guidelines", "memo": "memo", "readout": "readout"}


@component
class FilterExtractor:
    """Metadata filtering: "what did the research say about price?" should only look at research documents, and
    "how did Q3 go?" only at Q3 documents. Searching less is faster AND removes wrong answers that merely sound similar.
    Live: Claude chooses from the values that really exist. Offline: words in the question are matched to those values.
    `filters` is None when the question names nothing to filter on."""

    SYSTEM = ("You turn a question into metadata filters. Reply with ONLY JSON like "
              '{"doc_type": <one allowed value or null>, "period": <one allowed value or null>}. '
              "Use a value only if the question clearly asks for that kind of document or time period.")

    def __init__(self, values: dict[str, list[str]]):
        self.values = values

    @component.output_types(filters=dict | None, why=str, was_standin=bool)
    def run(self, query: str):
        reply = ask(self.SYSTEM, f"Allowed values: {json.dumps(self.values)}\nQuestion: {query}", "filter", max_tokens=60, offline="")
        chosen, standin = {}, not reply
        if reply:
            try:
                data = json.loads(re.search(r"\{.*\}", reply, re.S).group(0))
                chosen = {f: v for f, v in data.items() if f in self.values and v in self.values[f]}
            except Exception:
                standin = True
        if standin:
            chosen = self._rules(query)
        conds = [{"field": f"meta.{f}", "operator": "==", "value": v} for f, v in chosen.items()]
        filters = None if not conds else (conds[0] if len(conds) == 1 else {"operator": "AND", "conditions": conds})
        why = ", ".join(f"{f} = {v}" for f, v in chosen.items()) or "nothing in the question to filter on"
        return {"filters": filters, "why": why, "was_standin": standin}

    def _rules(self, query: str) -> dict:
        out = {}
        low = query.lower()
        m = re.search(r"\b(h[12]|q[1-4])\b(?:\s*(20\d\d))?", low)
        if m:
            period = f"{m.group(1).upper()} {m.group(2) or '2026'}"
            if period in self.values.get("period", []):
                out["period"] = period
        for word, doc_type in DOC_TYPE_WORDS.items():
            if re.search(rf"\b{word}\b", low) and doc_type in self.values.get("doc_type", []):
                out["doc_type"] = doc_type
                break
        return out


# --------------------------------------------------------------------------------- iterative retrieval

@component
class GapFinder:
    """Iterative (multi-step) retrieval: read what the first search found and say what is STILL MISSING, as a new search.
    Live: Claude names the gap. Offline: the question's words that no retrieved chunk contains."""

    SYSTEM = ("You help answer a question in several searches. Given the question and the text found so far, say what "
              "information is still missing, as ONE new search query. If the text already answers it, reply NONE.")

    @component.output_types(query=str | None, was_standin=bool)
    def run(self, question: str, documents: list[Document]):
        found = "\n".join(d.content[:300] for d in documents[:6])
        reply = ask(self.SYSTEM, f"Question: {question}\n\nFound so far:\n{found}", "gap", max_tokens=60, offline="")
        if reply:
            return {"query": None if reply.strip().upper().startswith("NONE") else reply.strip(), "was_standin": False}
        have = {t for d in documents for t in terms(d.content)}
        missing = [t for t in terms(question) if t not in have]
        return {"query": " ".join(missing) if missing else None, "was_standin": True}
