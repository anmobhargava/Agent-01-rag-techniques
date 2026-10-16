"""
Training data for fine-tuning the embedding model. I do not type it: Claude writes it from the chunks, my code checks it.

  ANTHROPIC_API_KEY=... python finetune/pairs.py            writes finetune/pairs.json

For every chunk, Claude writes 3 questions a person might really ask whose answer is in that chunk, in different styles
(short keywords, a natural question, a vague one). Each pair is (question, chunk). What my code checks, and drops on failure:
  - the question is a real question-length string (4 to 25 words)
  - it does not copy a long run of words straight out of the chunk (that would teach the model to match exact words)
  - no duplicate questions

Why this helps at all: a general model has never seen YOUR documents' vocabulary ("Crunch Time", "Value Shoppers", "Leaf Local").
Fine-tuning on pairs like these pulls those phrases towards the chunks that talk about them.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_agent.components import _json_list                       # noqa: E402
from rag_agent.index import load_chunks                           # noqa: E402
from rag_agent.llm import STRONG, ask, have_key                   # noqa: E402

SYSTEM = ("You write search questions for training a retrieval model. Given one text chunk, write 3 different questions a person "
          "might type whose answer is in that chunk: one as short keywords, one as a natural question, one vague or indirect. "
          "Do not copy long phrases from the chunk. Reply with ONLY a JSON list of 3 strings.")


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def copies_chunk(question: str, chunk: str, run: int = 7) -> bool:
    """True if the question contains `run` or more words in a row that appear in the chunk in the same order."""
    q, c = words(question), " ".join(words(chunk))
    return any(" ".join(q[i:i + run]) in c for i in range(max(0, len(q) - run + 1)))


def validate(question, chunk: str) -> str | None:
    if not isinstance(question, str):
        return "not a string"
    n = len(words(question))
    if not 4 <= n <= 25 and not (2 <= n <= 25 and "?" not in question):      # keyword-style queries may be shorter
        return "length"
    if copies_chunk(question, chunk):
        return "copies the chunk"
    return None


def make_pairs(chunks) -> tuple[list[dict], dict]:
    pairs, dropped, seen = [], {}, set()
    for c in chunks:
        for q in _json_list(ask(SYSTEM, c.content, "finetune_pairs", model=STRONG, max_tokens=250)):
            why = validate(q, c.content)
            if why is None and q.strip().lower() in seen:
                why = "duplicate"
            if why:
                dropped[why] = dropped.get(why, 0) + 1
                continue
            seen.add(q.strip().lower())
            pairs.append({"question": q.strip(), "chunk_id": c.meta["chunk_id"], "doc_id": c.meta["doc_id"]})
    return pairs, dropped


def main():
    if not have_key():
        sys.exit("No ANTHROPIC_API_KEY set. Fine-tuning data is written by Claude, so a key is required.")
    chunks = load_chunks()
    pairs, dropped = make_pairs(chunks)
    out = ROOT / "finetune" / "pairs.json"
    out.write_text(json.dumps({"generated_by": f"Claude ({STRONG})", "pairs": pairs}, indent=1))
    print(f"wrote {len(pairs)} pairs for {len(chunks)} chunks to {out.relative_to(ROOT)}; dropped: {dropped}")
    print("Next: skim the file, then `python finetune/train.py`.")


if __name__ == "__main__":
    main()
