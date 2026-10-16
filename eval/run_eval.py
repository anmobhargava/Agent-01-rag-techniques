"""
Score every technique, and the agent, on the generated questions.

  python eval/run_eval.py                       retrieval scores per technique and per question kind
  python eval/run_eval.py --ragas               also RAGAS on the agent's written answers (needs a Claude key; costs money)
  python eval/run_eval.py --questions eval/questions.smoke.json     wiring check on the mechanical questions

Reading the table: each cell is recall@k (share of the correct chunks that were found in the top k).
The "agent" row is the whole loop (plan, retrieve, grade, switch). It is scored on the evidence it kept, which can be
fewer than k chunks, so it trades a little recall for cleaner evidence. Compare it with "hybrid" first: the agent
is only worth its extra calls if it beats the plain default somewhere.

If no Claude key is set, the planner, grader and many techniques run as offline stand-ins. The numbers then describe
the stand-ins, and the table says so at the top.
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")
os.environ.setdefault("TRACE_LOG", "0")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_agent import agent, tracing                         # noqa: E402
from rag_agent.index import Index                            # noqa: E402
from rag_agent.llm import have_key                           # noqa: E402
from rag_agent.metrics import score                          # noqa: E402
from rag_agent.techniques import TECHNIQUES                  # noqa: E402


def mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def evaluate_retrieval(ix: Index, questions: list[dict], k: int) -> dict:
    """rows[name][kind] = list of per-question score dicts. `name` is a technique key or 'agent'."""
    rows: dict[str, dict[str, list]] = {}
    errors: dict[str, int] = {}
    for q in questions:
        for key, t in TECHNIQUES.items():
            try:
                docs = t.fn(ix, q["question"], k).documents
            except Exception:
                errors[key] = errors.get(key, 0) + 1
                docs = []
            rows.setdefault(key, {}).setdefault(q["kind"], []).append(score(docs, q["gold"], ix.by_id))
        with tracing.request_trace("eval", question=q["question"][:100]) as tr:
            ans = agent.run(q["question"], ix, k=k)
        docs = [d for d in _evidence_docs(ix, ans)]
        s = score(docs, q["gold"], ix.by_id)
        s["technique"], s["cost"] = ans.technique, ans.cost_usd
        rows.setdefault("agent", {}).setdefault(q["kind"], []).append(s)
    return {"rows": rows, "errors": errors}


def _evidence_docs(ix: Index, ans: agent.Answer):
    """Turn the agent's sources back into scorable objects (id + members + text)."""
    from haystack import Document
    return [Document(content=s["text"], meta={"chunk_id": s["id"], "members": s["members"]}) for s in ans.sources]


def print_table(result: dict, kinds: list[str]) -> None:
    head = f"{'technique':<14}" + "".join(f"{k:>11}" for k in kinds) + f"{'overall':>10}{'MRR':>8}"
    print(head)
    print("-" * len(head))
    order = list(TECHNIQUES) + ["agent"]
    for name in order:
        by_kind = result["rows"].get(name, {})
        cells, allr, allrr = [], [], []
        for k in kinds:
            sc = by_kind.get(k, [])
            cells.append(f"{mean([s['recall'] for s in sc]):>11}" if sc else f"{'-':>11}")
            allr += [s["recall"] for s in sc]
            allrr += [s["rr"] for s in sc]
        print(f"{name:<14}" + "".join(cells) + f"{mean(allr):>10}{mean(allrr):>8}")
    if result["errors"]:
        print("\ntechniques that raised errors (counted as no result):", result["errors"])
    chosen: dict[str, int] = {}
    for sc in result["rows"].get("agent", {}).values():
        for s in sc:
            chosen[s["technique"] or "(blocked)"] = chosen.get(s["technique"] or "(blocked)", 0) + 1
    print("\nagent's first-pick technique counts:", chosen)


def ragas_scores(ix: Index, questions: list[dict], limit: int) -> dict:
    """RAGAS on the agent's written answers: faithfulness (is the answer supported by the chunks it used?),
    context precision (are the useful chunks ranked first?) and context recall (do the chunks contain what the
    reference answer says?). Needs a key and costs money, so it takes a --limit. I have NOT run it yet."""
    import anthropic
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness
    llm = llm_factory("claude-sonnet-4-6", provider="anthropic", client=anthropic.Anthropic())
    faith, prec, rec = Faithfulness(llm=llm), ContextPrecision(llm=llm), ContextRecall(llm=llm)
    out: dict[str, dict[str, list]] = {}
    for q in questions[:limit]:
        ans = agent.run(q["question"], ix)
        ctx = [s["text"] for s in ans.sources]
        if not ctx:
            continue
        scores = {
            "faithfulness": faith.score(user_input=q["question"], response=ans.text, retrieved_contexts=ctx).value,
            "context_precision": prec.score(user_input=q["question"], reference=q["reference_answer"], retrieved_contexts=ctx).value,
            "context_recall": rec.score(user_input=q["question"], retrieved_contexts=ctx, reference=q["reference_answer"]).value,
        }
        for name, v in scores.items():
            out.setdefault(q["kind"], {}).setdefault(name, []).append(v)
    return {kind: {m: mean(v) for m, v in ms.items()} for kind, ms in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.json"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ragas", action="store_true")
    ap.add_argument("--ragas-limit", type=int, default=12)
    args = ap.parse_args()
    path = Path(args.questions)
    if not path.exists():
        sys.exit(f"{path} not found. Run `python eval/make_eval.py` (needs a key) or `--smoke` first.")
    data = json.loads(path.read_text())
    questions = data["questions"]
    kinds = [k for k in ("lookup", "context", "multi_hop", "broad") if any(q["kind"] == k for q in questions)]
    print(f"questions: {len(questions)} from {path.name} ({data.get('generated_by', '?')})")
    if not have_key():
        print("NOTE: no Claude key. Planner, grader and several techniques ran as offline stand-ins, so these numbers\n"
              "describe the stand-ins, not the real techniques.\n")
    ix = Index()
    result = evaluate_retrieval(ix, questions, args.k)
    print_table(result, kinds)
    report = {"questions": len(questions), "k": args.k, "live": have_key(),
              "recall_by_technique": {n: {k: mean([s["recall"] for s in sc]) for k, sc in by.items()} for n, by in result["rows"].items()}}
    if args.ragas:
        if not have_key():
            sys.exit("--ragas needs ANTHROPIC_API_KEY.")
        report["ragas"] = ragas_scores(ix, questions, args.ragas_limit)
        print("\nRAGAS (agent answers):", json.dumps(report["ragas"], indent=1))
    (ROOT / "eval" / "report.json").write_text(json.dumps(report, indent=1))
    print("\nsaved eval/report.json")


if __name__ == "__main__":
    main()
