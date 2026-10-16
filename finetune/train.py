"""
Fine-tune an embedding model on the generated pairs, and measure whether it actually got better.

  python finetune/train.py                       base model BAAI/bge-small-en-v1.5 -> models/crunchwell-bge-small
  python finetune/train.py --base some/model --epochs 3
  EMBEDDER=st:models/crunchwell-bge-small python -m rag_agent.app       use it in the app

The honest way to fine-tune, in five steps:
  1. SPLIT BY DOCUMENT. Pairs from some documents train the model, pairs from the other documents are held out. If I split by
     pair instead, the model would see the same chunk in training and in the test, and the "improvement" would be memorisation.
  2. MEASURE BEFORE. Recall@5 and MRR of the untouched base model on the held-out pairs, searching ALL chunks.
  3. MINE HARD NEGATIVES. For each training pair, find a chunk the base model wrongly ranks near the top. Training against
     those is what teaches the model the difference between "Crunch Time results" and "Leaf Local results".
  4. TRAIN with MultipleNegativesRankingLoss on (question, right chunk, hard negative).
  5. MEASURE AFTER on the same held-out pairs, and print both.

What to expect: with 63 chunks (about 190 pairs) this is a TINY dataset. It shows the pipeline and can nudge the numbers, but a
gain here is weak evidence, and a loss means overfitting. Real fine-tuning wants thousands of pairs. Do not quote a number from
this script as a result without saying how small the data is.

Needs requirements-finetune.txt (torch, sentence-transformers). I have NOT run it with a real base model yet
(no model download was possible). The training loop is exercised with a tiny random model in tests/test_finetune.py when torch is installed.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_agent.index import load_chunks                            # noqa: E402


def split_by_doc(pairs: list[dict], holdout: float = 0.3) -> tuple[list[dict], list[dict]]:
    """Hold out whole documents. A document goes to the held-out side when a stable hash of its id falls under `holdout`,
    so the split never changes between runs and never depends on the order of the file."""
    def held(doc_id: str) -> bool:
        return int(hashlib.sha1(doc_id.encode()).hexdigest(), 16) % 1000 < holdout * 1000
    train = [p for p in pairs if not held(p["doc_id"])]
    test = [p for p in pairs if held(p["doc_id"])]
    return train, test


def evaluate(encode_docs, encode_queries, chunks, pairs: list[dict], k: int = 5) -> dict:
    """Recall@k and MRR. `encode_docs(list[str]) -> array` and `encode_queries(list[str]) -> array` are the model under test.
    For each pair the right answer is its chunk; every chunk in the corpus is a candidate."""
    ids = [c.meta["chunk_id"] for c in chunks]
    d = np.asarray(encode_docs([c.content for c in chunks]))
    q = np.asarray(encode_queries([p["question"] for p in pairs]))
    order = np.argsort(-(q @ d.T), axis=1)
    hits, rr = 0, []
    for row, p in zip(order, pairs):
        rank = [ids[i] for i in row].index(p["chunk_id"]) + 1
        hits += rank <= k
        rr.append(1 / rank)
    return {f"recall@{k}": round(hits / len(pairs), 3), "mrr": round(float(np.mean(rr)), 3), "n": len(pairs)}


def mine_negatives(encode_docs, encode_queries, chunks, pairs: list[dict], window: int = 5) -> list[tuple[str, str, str]]:
    """(question, right chunk text, wrong-but-close chunk text). The wrong chunk is the best-ranked chunk that is NOT the right one
    and NOT from the same document (a sibling section of the same document may legitimately also answer, so it is not safe to call it wrong)."""
    ids = [c.meta["chunk_id"] for c in chunks]
    by_id = {c.meta["chunk_id"]: c for c in chunks}
    d = np.asarray(encode_docs([c.content for c in chunks]))
    q = np.asarray(encode_queries([p["question"] for p in pairs]))
    order = np.argsort(-(q @ d.T), axis=1)
    triples = []
    for row, p in zip(order, pairs):
        for i in row[: window + 5]:
            other = by_id[ids[i]]
            if other.meta["doc_id"] != p["doc_id"]:
                triples.append((p["question"], by_id[p["chunk_id"]].content, other.content))
                break
    return triples


def train(base: str, triples: list[tuple[str, str, str]], out_dir: Path, epochs: int = 2, batch_size: int = 16, lr: float = 2e-5,
          model=None):
    """MultipleNegativesRankingLoss on (anchor, positive, negative). Every OTHER positive in the batch also acts as a negative
    for free (in-batch negatives), so a bigger batch is a harder, better exercise. `model` lets a test pass a tiny model."""
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    model = model or SentenceTransformer(base)
    steps = epochs * -(-len(triples) // batch_size)
    warmup = max(1, steps // 10)                      # 10% of the steps; an int works on every transformers version
    data = Dataset.from_dict({"anchor": [t[0] for t in triples], "positive": [t[1] for t in triples], "negative": [t[2] for t in triples]})
    args = SentenceTransformerTrainingArguments(output_dir=str(out_dir / "checkpoints"), num_train_epochs=epochs,
                                                per_device_train_batch_size=batch_size, learning_rate=lr, warmup_steps=warmup,
                                                logging_steps=10, save_strategy="no", report_to="none", seed=42)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=data, loss=MultipleNegativesRankingLoss(model)).train()
    model.save(str(out_dir))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--pairs", default=str(ROOT / "finetune" / "pairs.json"))
    ap.add_argument("--out", default=str(ROOT / "models" / "crunchwell-bge-small"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--holdout", type=float, default=0.3)
    args = ap.parse_args()
    path = Path(args.pairs)
    if not path.exists():
        sys.exit(f"{path} not found. Run `python finetune/pairs.py` first (needs a Claude key).")
    from sentence_transformers import SentenceTransformer
    pairs = json.loads(path.read_text())["pairs"]
    chunks = load_chunks()
    train_pairs, test_pairs = split_by_doc(pairs, args.holdout)
    print(f"{len(train_pairs)} training pairs, {len(test_pairs)} held-out pairs (split by document)")
    if not train_pairs or not test_pairs:
        sys.exit("The document split left one side empty. Generate more pairs or change --holdout.")

    base = SentenceTransformer(args.base)
    enc = lambda texts: base.encode(texts, normalize_embeddings=True)                       # noqa: E731
    before = evaluate(enc, enc, chunks, test_pairs)
    print("before:", before)
    triples = mine_negatives(enc, enc, chunks, train_pairs)
    print(f"{len(triples)} training triples with a hard negative each")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tuned = train(args.base, triples, out, args.epochs)
    enc2 = lambda texts: tuned.encode(texts, normalize_embeddings=True)                     # noqa: E731
    after = evaluate(enc2, enc2, chunks, test_pairs)
    print("after: ", after)
    report = {"base": args.base, "train_pairs": len(train_pairs), "test_pairs": len(test_pairs), "before": before, "after": after,
              "note": "tiny dataset; treat as a demonstration of the pipeline, not a benchmark"}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"saved model and report.json to {out}. Use it with EMBEDDER=st:{out}")


if __name__ == "__main__":
    main()
