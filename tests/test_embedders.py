"""Embedding layer: the stand-in, the choice logic, the real-model wiring (with fakes), MaxSim and late chunking."""
import sys
import types

import numpy as np
import pytest

from rag_agent import embedders
from rag_agent.embedders import (LsaEmbedder, ToyContextEncoder, get_embedder, late_chunk, maxsim, token_spans, unit)

CORPUS = ["Spend was 6.2 million dollars on the creator campaign.", "Retail media prices rose through August.",
          "Search returns flatten as monthly spend rises.", "Gen Z discovers products on TikTok creators.",
          "Value Shoppers read shelf prices and multipacks closely.", "Television effects last about three weeks."]


def test_lsa_embedder_gives_unit_vectors_and_per_word_vectors():
    e = LsaEmbedder(CORPUS, dims=4)
    v = e.docs(CORPUS)
    assert v.shape == (6, e.dims) and np.allclose(np.linalg.norm(v, axis=1), 1)
    toks = e.tokens(["creator campaign spend"])[0]
    assert toks.shape[1] == e.dims and 1 <= len(toks) <= 3
    assert e.tokens(["zzzz unknownword"])[0].shape == (1, e.dims)                  # nothing known -> one zero vector, not a crash
    assert not e.real and "not neural" in e.status


def test_maxsim_adds_up_the_best_match_per_query_token():
    q = np.array([[1.0, 0], [0, 1.0]])
    d = np.array([[1.0, 0], [0.6, 0.8]])
    assert maxsim(q, d) == pytest.approx(1.0 + 0.8)                                 # token 1 -> 1.0, token 2 -> 0.8
    assert maxsim(q, np.array([[0.0, 0.0]])) == 0.0


def test_get_embedder_choices(monkeypatch):
    assert isinstance(get_embedder(CORPUS, "lsa"), LsaEmbedder)
    monkeypatch.setattr(embedders, "FastEmbedEmbedder", lambda *a, **k: (_ for _ in ()).throw(OSError("no network")))
    fallback = get_embedder(CORPUS, "auto")                                         # auto: falls back and says why
    assert isinstance(fallback, LsaEmbedder) and "OSError" in fallback.fallback_reason
    with pytest.raises(OSError):
        get_embedder(CORPUS, "fastembed")                                           # explicit: fail loudly


def test_get_embedder_reads_the_environment_and_st_prefix(monkeypatch):
    monkeypatch.setenv("EMBEDDER", "lsa")
    assert isinstance(get_embedder(CORPUS), LsaEmbedder)
    seen = {}
    monkeypatch.setattr(embedders, "SentenceTransformerEmbedder", lambda path: seen.setdefault("path", path) or "st")
    get_embedder(CORPUS, "st:models/mine")
    assert seen["path"] == "models/mine"


def test_fastembed_wiring_with_a_fake_library(monkeypatch):
    """No model can be downloaded here, so this only checks that I call fastembed the way it expects."""
    class FakeText:
        def __init__(self, model_name):
            self.model_name = model_name

        def embed(self, texts):
            return (np.ones(3) for _ in texts)

        def passage_embed(self, texts):
            return (np.array([1.0, 2.0, 2.0]) for _ in texts)

        def query_embed(self, text):
            return iter([np.array([2.0, 0.0, 0.0])])

    class FakeColbert:
        def __init__(self, model_name):
            pass

        def passage_embed(self, texts):
            return (np.ones((4, 2)) for _ in texts)

        def query_embed(self, texts):
            return (np.ones((2, 2)) for _ in texts)

    fake = types.SimpleNamespace(TextEmbedding=FakeText, LateInteractionTextEmbedding=FakeColbert)
    monkeypatch.setitem(sys.modules, "fastembed", fake)
    e = embedders.FastEmbedEmbedder("fake/model")
    assert e.real and e.dims == 3 and "neural" in e.status
    assert np.allclose(np.linalg.norm(e.docs(["a", "b"]), axis=1), 1)
    assert np.allclose(e.query("q"), [1, 0, 0])
    assert e.tokens(["x"])[0].shape == (4, 2) and e.tokens(["x"], is_query=True)[0].shape == (2, 2)


def test_token_spans_skip_special_tokens_and_map_characters_to_tokens():
    offsets = [(0, 0), (0, 5), (6, 11), (12, 16), (0, 0)]                           # [CLS] one two three [SEP]
    assert token_spans(offsets, [(0, 11), (12, 16), (50, 60)]) == [(1, 3), (3, 4), (0, 0)]


def test_late_chunk_averages_each_chunks_own_tokens_and_normalises():
    toks = np.array([[1.0, 0], [3.0, 0], [0, 2.0], [0, 4.0]])
    out = late_chunk(toks, [(0, 2), (2, 4), (0, 0)])
    assert np.allclose(out[0], [1, 0]) and np.allclose(out[1], [0, 1]) and np.allclose(out[2], [0, 0])


def test_toy_late_encoder_lets_a_chunk_see_its_neighbours():
    """The one property late chunking relies on: a chunk's vector is influenced by text OUTSIDE the chunk."""
    base = LsaEmbedder(CORPUS, dims=4)
    text = "Creator campaign spend was large. Television effects last long. Retail media prices rose."
    first = (0, text.index("Television"))
    late = ToyContextEncoder(base, window=6).encode_doc(text, [first])[0]
    alone = base.docs([text[first[0]:first[1]]])[0]
    assert not np.allclose(late, alone)                                             # context changed the vector
    far = ToyContextEncoder(base, window=0).encode_doc(text, [first])[0]            # window 0 = no context mixing
    other = base.term_vectors[base.vec.vocabulary_["television"]]
    assert float(late @ other) > float(far @ other)                                 # and it moved TOWARDS the next chunk's topic
