"""
Make the evaluation questions. I do NOT type them: Claude writes them from the documents and my code checks them.

  ANTHROPIC_API_KEY=... python eval/make_eval.py            writes eval/questions.json   (the real eval set)
  python eval/make_eval.py --smoke                          writes eval/questions.smoke.json (no key; wiring check only)

Four kinds of question, because techniques win on different kinds:
  lookup     one specific fact in one chunk
  context    one fact in one chunk, asked by campaign or audience name, though the chunk itself never says the name
  multi_hop  needs facts from two or more chunks in different documents
  broad      themes or lessons across the documents (three or more chunks)

What my code checks on everything Claude writes (a question is dropped if any check fails):
  - every cited chunk id exists
  - every evidence quote appears word for word in the chunk it is attached to
  - the number of chunks fits the kind (lookup and context: exactly 1; multi_hop: 2+ from 2+ documents; broad: 3+)

Then I read eval/questions.json once, and delete or fix anything odd. That review is the human part.
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_agent.components import terms                   # noqa: E402
from rag_agent.index import load_chunks                              # noqa: E402
from rag_agent.llm import STRONG, ask, have_key                      # noqa: E402

KINDS = {
    "lookup": "a specific fact (a number, name or date) that is stated in ONE chunk",
    "context": ("a specific fact in ONE chunk, but the question names the campaign or audience it is about, "
                "even though the chunk text itself does not name it (for example a 'Results' or 'Learnings' section)"),
    "multi_hop": "a question that needs facts from TWO OR MORE chunks that come from DIFFERENT documents",
    "broad": "a broad question about themes, lessons or patterns that needs THREE OR MORE chunks",
}
SYSTEM = ("You write evaluation questions for a retrieval system, using only the chunks you are shown. "
          "Reply with ONLY a JSON list. Each item: {\"question\": str, \"reference_answer\": str (1-2 sentences), "
          "\"evidence\": [{\"chunk_id\": str, \"quote\": str}]}. Every quote must be copied word for word from the chunk it is attached to.")


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def validate(item: dict, kind: str, by_id: dict) -> str | None:
    """Return None if the item is good, else the reason it was dropped."""
    if not isinstance(item, dict) or not isinstance(item.get("question"), str) or not item["question"].strip():
        return "no question"
    if not isinstance(item.get("reference_answer"), str) or not item["reference_answer"].strip():
        return "no reference answer"
    ev = item.get("evidence")
    if not isinstance(ev, list) or not ev:
        return "no evidence"
    ids = []
    for e in ev:
        if not isinstance(e, dict) or e.get("chunk_id") not in by_id:
            return "unknown chunk id"
        quote = squash(str(e.get("quote", "")))
        if len(quote) < 8 or quote not in squash(by_id[e["chunk_id"]].content):
            return "quote not found in chunk"
        ids.append(e["chunk_id"])
    unique = set(ids)
    docs = {i.split("/")[0] for i in unique}
    if kind in ("lookup", "context") and len(unique) != 1:
        return "needs exactly 1 chunk"
    if kind == "multi_hop" and (len(unique) < 2 or len(docs) < 2):
        return "needs 2+ chunks from 2+ documents"
    if kind == "broad" and len(unique) < 3:
        return "needs 3+ chunks"
    return None


def generate(n_per_kind: int, chunks) -> tuple[list[dict], dict]:
    by_id = {c.meta["chunk_id"]: c for c in chunks}
    corpus = "\n\n".join(f"[{c.meta['chunk_id']}]\n{c.content}" for c in chunks)
    out, dropped = [], {}
    for kind, desc in KINDS.items():
        reply = ask(SYSTEM, f"Chunks:\n\n{corpus}\n\nWrite {n_per_kind} different questions. Each one is {desc}. "
                            "Spread them across different documents.", f"make_eval_{kind}",
                    model=STRONG, max_tokens=3500)
        items = _json_list_of_dicts(reply)
        for item in items:
            why = validate(item, kind, by_id)
            if why:
                dropped[why] = dropped.get(why, 0) + 1
                continue
            out.append({"id": f"{kind}-{sum(1 for o in out if o['kind'] == kind) + 1}", "kind": kind,
                        "question": item["question"].strip(), "reference_answer": item["reference_answer"].strip(),
                        "gold": sorted({e["chunk_id"] for e in item["evidence"]}), "evidence": item["evidence"]})
    return out, dropped


def _json_list_of_dicts(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)]


def smoke(chunks) -> list[dict]:
    """No model: for each chunk, a query made of its 5 rarest content words. Only checks that the harness runs.
    Keyword queries like this flatter BM25, so these numbers say nothing about which technique is better."""
    df: dict[str, int] = {}
    for c in chunks:
        for t in set(terms(c.content)):
            df[t] = df.get(t, 0) + 1
    out = []
    for c in chunks:
        rare = sorted(set(terms(c.content)), key=lambda t: (df[t], t))[:5]
        out.append({"id": f"smoke-{len(out) + 1}", "kind": "lookup", "question": " ".join(rare),
                    "reference_answer": "", "gold": [c.meta["chunk_id"]], "evidence": []})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="mechanical keyword questions, no Claude key needed")
    ap.add_argument("--per-kind", type=int, default=8)
    args = ap.parse_args()
    chunks = load_chunks()
    if args.smoke:
        items, path, meta = smoke(chunks), ROOT / "eval" / "questions.smoke.json", {"generated_by": "smoke: mechanical keywords, wiring check only"}
        dropped = {}
    else:
        if not have_key():
            sys.exit("No ANTHROPIC_API_KEY set. Use --smoke for a wiring check, or set a key to generate the real set.")
        items, dropped = generate(args.per_kind, chunks)
        path, meta = ROOT / "eval" / "questions.json", {"generated_by": f"Claude ({STRONG})", "date": date.today().isoformat()}
    path.write_text(json.dumps({**meta, "questions": items}, indent=1))
    kinds = {k: sum(1 for i in items if i["kind"] == k) for k in KINDS}
    print(f"wrote {len(items)} questions to {path.relative_to(ROOT)}  {kinds}")
    if dropped:
        print("dropped by my checks:", dropped)
    print("Next: read the file once and delete or fix anything odd.")


if __name__ == "__main__":
    main()
