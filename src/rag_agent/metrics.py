"""
Retrieval scores. Three plain numbers per question:
  hit     did at least one returned item cover a correct chunk?      (1 or 0)
  recall  what share of the correct chunks were covered?             (0 to 1)
  rr      1 / rank of the first item that covers a correct chunk     (0 if none)  -> averaged this is MRR

"Covers" is not just "same id". Some techniques return other things than the original chunks: a parent piece,
a summary of several chunks, a community report. So an item covers a correct chunk when ANY of these is true:
  - it has the same id,
  - the correct chunk is listed among the chunks it summarises (its `members`),
  - it contains at least 60% of the correct chunk's words (a bigger piece that includes it).
"""

import re


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def covers(doc, gold_id: str, by_id: dict) -> bool:
    if doc.meta.get("chunk_id") == gold_id or gold_id in doc.meta.get("members", ()):
        return True
    gold = by_id.get(gold_id)
    if gold is None:
        return False
    g = words(gold.content)
    return bool(g) and len(g & words(doc.content)) / len(g) >= 0.6


def score(docs: list, gold_ids: list[str], by_id: dict) -> dict:
    covered = {g for g in gold_ids if any(covers(d, g, by_id) for d in docs)}
    rr = 0.0
    for rank, d in enumerate(docs, 1):
        if any(covers(d, g, by_id) for g in gold_ids):
            rr = 1 / rank
            break
    return {"hit": 1.0 if covered else 0.0, "recall": len(covered) / len(gold_ids) if gold_ids else 0.0, "rr": rr}
