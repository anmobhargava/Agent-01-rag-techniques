"""
Where text becomes numbers. Everything that needs vectors asks for an Embedder, so swapping the model is one setting.

  FastEmbedEmbedder          a real neural model (default BAAI/bge-small-en-v1.5, runs on CPU through ONNX)
  SentenceTransformerEmbedder  your own fine-tuned model, loaded from a folder (see finetune_embeddings.py)
  LsaEmbedder                TF-IDF + SVD. NOT neural. It exists so the repo still runs where no model can be downloaded.

Pick with the EMBEDDER setting:
  auto (default)   try the neural model, fall back to LsaEmbedder if it cannot load, and say so
  lsa              always the stand-in (the tests use this so they never touch the network)
  fastembed        the neural model, and fail loudly if it cannot load
  st:<folder>      a SentenceTransformer folder, e.g. st:models/crunchwell-bge-small

Three different things use vectors, so an Embedder can give three kinds:
  docs()/query()   ONE vector per text (normal dense search)
  tokens()         ONE VECTOR PER TOKEN (ColBERT-style late interaction, scored with MaxSim)
  late chunking    see the functions at the bottom: embed the whole document first, then cut it into chunk vectors

I have not run the neural paths with real weights yet. I checked the wiring with fakes, because I could not download
a model while building this. The stand-in paths run for real.
"""

import os
import re

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_COLBERT = os.environ.get("COLBERT_MODEL", "colbert-ir/colbertv2.0")


def unit(x: np.ndarray) -> np.ndarray:
    """Scale vectors to length 1, so a dot product IS cosine similarity."""
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1, n)


class Embedder:
    name = "base"
    real = False                      # True only for a neural model
    dims = 0

    def docs(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def query(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def tokens(self, texts: list[str], is_query: bool = False) -> list[np.ndarray]:
        raise NotImplementedError

    @property
    def status(self) -> str:
        return f"neural model: {self.name}" if self.real else f"stand-in (not neural): {self.name}"


# ------------------------------------------------------------------------------------------------ the stand-in

class LsaEmbedder(Embedder):
    """TF-IDF followed by SVD gives each text a short dense vector. It captures which words go together in THIS
    corpus, so it is a real vector search, but it cannot match 'cheap' with 'low-cost' the way a trained model does.
    For late interaction it uses each WORD's column of the SVD as that word's vector."""

    name = "TF-IDF + SVD"

    def __init__(self, corpus: list[str], dims: int = 32):
        self.vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        x = self.vec.fit_transform(corpus)
        self.dims = max(2, min(dims, x.shape[0] - 1, x.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=self.dims, random_state=0).fit(x)
        self.term_vectors = unit(self.svd.components_.T)                   # one row per vocabulary word
        self.analyzer = self.vec.build_analyzer()

    def docs(self, texts):
        return unit(self.svd.transform(self.vec.transform(texts)))

    def query(self, text):
        return self.docs([text])[0]

    def tokens(self, texts, is_query=False):
        out = []
        for t in texts:
            idx = [self.vec.vocabulary_[w] for w in self.analyzer(t) if w in self.vec.vocabulary_]
            out.append(self.term_vectors[idx] if idx else np.zeros((1, self.dims)))
        return out


# ------------------------------------------------------------------------------------------------- the real ones

class FastEmbedEmbedder(Embedder):
    """A real embedding model through fastembed (ONNX, CPU, no torch). The model is downloaded on first use."""

    real = True

    def __init__(self, model_name: str = DEFAULT_MODEL, colbert_name: str = DEFAULT_COLBERT):
        from fastembed import TextEmbedding
        self.name = model_name
        self.model = TextEmbedding(model_name=model_name)
        self.dims = len(next(iter(self.model.embed(["probe"]))))
        self._colbert_name, self._colbert = colbert_name, None

    def docs(self, texts):
        return unit(np.array(list(self.model.passage_embed(texts))))

    def query(self, text):
        return unit(np.array(next(iter(self.model.query_embed(text)))))

    def tokens(self, texts, is_query=False):
        """ColBERT: one vector per token. A separate model, loaded only if late interaction is used."""
        if self._colbert is None:
            from fastembed import LateInteractionTextEmbedding
            self._colbert = LateInteractionTextEmbedding(model_name=self._colbert_name)
        gen = self._colbert.query_embed(texts) if is_query else self._colbert.passage_embed(texts)
        return [unit(np.asarray(v)) for v in gen]


class SentenceTransformerEmbedder(Embedder):
    """Your own fine-tuned model (or any sentence-transformers model) from a folder. Needs requirements-finetune.txt."""

    real = True

    def __init__(self, path: str):
        from sentence_transformers import SentenceTransformer
        self.name = path
        self.model = SentenceTransformer(path)
        self.dims = self.model.get_sentence_embedding_dimension()

    def docs(self, texts):
        return unit(self.model.encode(texts, normalize_embeddings=True))

    def query(self, text):
        return unit(self.model.encode([text], normalize_embeddings=True)[0])

    def tokens(self, texts, is_query=False):
        out = self.model.encode(texts, output_value="token_embeddings")
        return [unit(np.asarray(o)) for o in out]


def get_embedder(corpus: list[str], choice: str | None = None) -> Embedder:
    """Choose the embedder from EMBEDDER (see the top of this file). `fallback_reason` records why a stand-in was used."""
    choice = (choice or os.environ.get("EMBEDDER", "auto")).strip()
    if choice == "lsa":
        return LsaEmbedder(corpus)
    if choice.startswith("st:"):
        return SentenceTransformerEmbedder(choice[3:])
    if choice == "fastembed":
        return FastEmbedEmbedder()
    try:                                                           # auto
        return FastEmbedEmbedder()
    except Exception as e:
        emb = LsaEmbedder(corpus)
        emb.fallback_reason = f"{type(e).__name__}: could not load {DEFAULT_MODEL}"
        return emb


# ----------------------------------------------------------------------------------- late interaction (ColBERT)

def maxsim(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """ColBERT's score. For each QUERY token, find its best-matching DOCUMENT token, then add those up.
    Compared with one vector per text, it lets 'Crunch' match 'Crunch' and 'spend' match 'spent' separately instead of
    squashing the whole text into a single point. The price is storing one vector per token."""
    sims = np.asarray(query_vecs) @ np.asarray(doc_vecs).T            # (query tokens, doc tokens)
    return float(sims.max(axis=1).sum())


# --------------------------------------------------------------------------------------------- late chunking

def token_spans(offsets: list[tuple[int, int]], char_spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Turn character ranges ('this chunk is characters 120 to 410') into token ranges, using the tokenizer's offsets.
    Special tokens have offset (0, 0) and are skipped."""
    out = []
    for cs, ce in char_spans:
        inside = [i for i, (s, e) in enumerate(offsets) if e > s and s >= cs and e <= ce]
        out.append((inside[0], inside[-1] + 1) if inside else (0, 0))
    return out


def late_chunk(token_vecs: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
    """Late chunking. Normal chunking cuts the text first and embeds each piece alone, so a piece like 'Spend was 6.2 million'
    knows nothing about its campaign. Here the WHOLE document goes through the model once, so every token vector has already
    'read' the rest of the document. Only then do I cut: each chunk vector is the average of its own tokens' vectors."""
    out = []
    for s, e in spans:
        out.append(token_vecs[s:e].mean(axis=0) if e > s else np.zeros(token_vecs.shape[1]))
    return unit(np.array(out))


class ToyContextEncoder:
    """OFFLINE STAND-IN for a long-context model. Each word's vector is mixed with its neighbours across the whole
    document (a moving average), so a token near a chunk's edge carries some of the next chunk's meaning. That is the one
    property late chunking relies on. It is a toy: the real thing needs a long-context model such as HFLateChunker."""

    name = "toy moving-average context (stand-in)"
    real = False

    def __init__(self, base: LsaEmbedder, window: int = 15):
        self.base, self.window, self.dims = base, window, base.dims

    def _doc_tokens(self, text: str):
        """Per-word vectors for the whole text, smoothed, plus each word's character offset."""
        words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z0-9]+", text)]
        vecs = []
        for w, _, _ in words:
            i = self.base.vec.vocabulary_.get(w.lower())
            vecs.append(self.base.term_vectors[i] if i is not None else np.zeros(self.dims))
        v = np.array(vecs) if vecs else np.zeros((1, self.dims))
        k = self.window
        smooth = np.array([v[max(0, i - k): i + k + 1].mean(axis=0) for i in range(len(v))])
        return smooth, [(s, e) for _, s, e in words]

    def encode_doc(self, text: str, char_spans: list[tuple[int, int]]) -> np.ndarray:
        vecs, offsets = self._doc_tokens(text)
        return late_chunk(vecs, token_spans(offsets, char_spans))

    def query(self, text: str) -> np.ndarray:
        return self.base.query(text)


class HFLateChunker:
    """Late chunking with a real long-context model (default jina-embeddings-v2-small-en, 8k tokens).
    Needs requirements-finetune.txt (torch + transformers). I have NOT run it, so treat it as
    written-but-unverified until it has run once."""

    real = True

    def __init__(self, model_name: str = os.environ.get("LATE_MODEL", "jinaai/jina-embeddings-v2-small-en")):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.name, self.torch = model_name, torch
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()
        self.dims = self.model.config.hidden_size

    def _tokens(self, text: str):
        enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=8192)
        offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
        with self.torch.no_grad():
            hidden = self.model(**enc).last_hidden_state[0].numpy()
        return hidden, offsets

    def encode_doc(self, text: str, char_spans: list[tuple[int, int]]) -> np.ndarray:
        hidden, offsets = self._tokens(text)
        return late_chunk(hidden, token_spans(offsets, char_spans))

    def query(self, text: str) -> np.ndarray:
        hidden, _ = self._tokens(text)
        return unit(hidden.mean(axis=0))
