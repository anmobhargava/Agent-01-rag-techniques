"""Fine-tuning data prep and evaluation run offline. The training loop needs torch, so it is skipped when torch is missing."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune"))
from pairs import copies_chunk, make_pairs, validate          # noqa: E402
from train import evaluate, mine_negatives, split_by_doc      # noqa: E402

CHUNK = "Crunch Time spent 6.2 million dollars across creators and short video in the third quarter."


def test_validate_rejects_copies_bad_lengths_and_non_strings():
    assert validate("crunch time spend", CHUNK) is None                                   # short keyword query is fine
    assert validate("Crunch Time spent 6.2 million dollars across creators and short video", CHUNK) == "copies the chunk"
    assert validate("hi", CHUNK) == "length" and validate(5, CHUNK) == "not a string"
    assert copies_chunk("it spent 6.2 million dollars across creators and short", CHUNK)
    assert not copies_chunk("how much money went into the creator push", CHUNK)


def test_make_pairs_uses_claude_output_and_drops_duplicates(offline_index, fake_claude):
    pairs, dropped = make_pairs(offline_index.chunks[:3])
    assert pairs and all({"question", "chunk_id", "doc_id"} <= set(p) for p in pairs)
    assert len({p["question"].lower() for p in pairs}) == len(pairs)
    assert dropped.get("duplicate", 0) >= 1                                              # the fake returns the same 3 questions for every chunk


def test_split_by_doc_never_puts_one_document_on_both_sides_and_is_stable():
    pairs = [{"doc_id": f"d{i}", "chunk_id": f"d{i}/a", "question": "q"} for i in range(40)] * 2
    train, test = split_by_doc(pairs)
    assert not {p["doc_id"] for p in train} & {p["doc_id"] for p in test} and train and test
    assert split_by_doc(pairs) == (train, test)


def test_evaluate_and_negatives_with_a_hand_made_encoder(offline_index):
    chunks = offline_index.chunks
    emb = offline_index.embedder
    pairs = [{"question": c.content, "chunk_id": c.meta["chunk_id"], "doc_id": c.meta["doc_id"]} for c in chunks[:8]]
    report = evaluate(emb.docs, lambda t: emb.docs(t), chunks, pairs, k=3)
    assert report["recall@3"] == 1.0 and report["mrr"] > 0.9 and report["n"] == 8        # a question that IS the chunk must be found
    wrong = lambda texts: -emb.docs(texts)                                               # noqa: E731  (an anti-model)
    assert evaluate(emb.docs, wrong, chunks, pairs, k=1)["recall@1"] < 1.0
    for q, pos, neg in mine_negatives(emb.docs, emb.docs, chunks, pairs):
        assert pos != neg and q


def test_training_loop_runs_on_a_tiny_random_model(tmp_path, offline_index):
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("datasets")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
    from sentence_transformers import SentenceTransformer, models as stm
    words = sorted({w for c in offline_index.chunks for w in c.content.lower().split()})
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, **{w: i + 4 for i, w in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")); tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")
    bert = BertModel(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32))
    (tmp_path / "base").mkdir(); bert.save_pretrained(tmp_path / "base"); fast.save_pretrained(tmp_path / "base")
    model = SentenceTransformer(modules=[stm.Transformer(str(tmp_path / "base")), stm.Pooling(16)])
    from train import train
    chunks = offline_index.chunks
    triples = [(c.content[:60], c.content, chunks[(i + 5) % len(chunks)].content) for i, c in enumerate(chunks[:8])]
    train("unused", triples, tmp_path / "out", epochs=1, batch_size=4, model=model)
    assert (tmp_path / "out" / "modules.json").exists()
    from rag_agent.embedders import get_embedder
    emb = get_embedder([], f"st:{tmp_path / 'out'}")                       # the saved folder loads the way EMBEDDER=st:<folder> does
    assert emb.real and emb.docs(["crunch time spend"]).shape == (1, 16) and abs(np.linalg.norm(emb.query("spend")) - 1) < 1e-5
