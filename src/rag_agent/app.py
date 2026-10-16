"""
The chat app. Run it with:   python -m rag_agent.app      (from the repo root, with PYTHONPATH=src)

Three tabs:
  Chat      ask a question. The right side shows how the agent decided (plan, retrieve, grade, switch, answer, verify)
            and the exact chunks it used. Pick a technique yourself in the dropdown, or leave it on "Let the agent choose".
  Compare   same question through all 16 techniques, side by side.
  About     what each technique is for, and what is real versus a stand-in in this run.

The agent runs in a background thread and pushes its steps into a queue. The chat reads the queue and
updates the screen as each step arrives, so you watch it work and the answer streams in.

Cost guard: with a live key, each browser session gets MAX_LIVE_QUESTIONS questions (default 15).
"""

import os
import queue
import threading

os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from . import agent, tracing
from .index import Index
from .llm import have_key
from .semantic_cache import SemanticCache
from .techniques import TECHNIQUES

MAX_LIVE_QUESTIONS = int(os.environ.get("MAX_LIVE_QUESTIONS", "15"))
EXAMPLES = [
    "How much did Crunch Time spend?",
    "What are the main lessons across all campaigns?",
    "Which agency ran Leaf Local and what is its fee?",
    "Why did price-sensitive households buy less after the price increase?",
    "Where should we move budget according to the marketing mix model?",
]
AUTO = "Let the agent choose"

_index: Index | None = None
_lock = threading.RLock()                 # get_cache calls get_index while holding it


def get_index() -> Index:
    """Build the index once, on the first question, not at import. Building can call Claude, so it is lazy."""
    global _index
    with _lock:
        if _index is None:
            _index = Index()
        return _index


_cache: SemanticCache | None = None


def get_cache() -> SemanticCache:
    global _cache
    with _lock:
        if _cache is None:
            ix = get_index()
            _cache = SemanticCache(ix.embedder, ix.corpus_hash)
        return _cache


def banner() -> str:
    if have_key():
        return "**Live.** Claude plans, grades, rewrites and writes the answers."
    return ("**Offline demo.** No Claude key is set, so planning uses rules, grading is a word-overlap check, and an "
            "answer is the best-matching sentences instead of a written reply. Set `ANTHROPIC_API_KEY` for the real thing.")


# ----------------------------------------------------------------------------------------- rendering

def render_steps(steps: list[dict]) -> str:
    if not steps:
        return "_The agent's steps will show up here._"
    return "\n\n".join(f"**{i}. {s['title']}**  \n{s['detail']}" for i, s in enumerate(steps, 1))


def render_sources(sources: list[dict]) -> str:
    if not sources:
        return "_No chunks used._"
    out = []
    for s in sources:
        extra = f" (summary of {len(s['members'])} chunks)" if len(s.get("members", [])) > 1 else ""
        body = s["text"].replace("\n", "  \n")
        out.append(f"<details><summary><code>{s['id']}</code>{extra}</summary>\n\n{body}\n\n</details>")
    return "\n\n".join(out)


def render_footer(a: agent.Answer) -> str:
    bits = [f"technique: **{TECHNIQUES[a.technique].label}**" if a.technique else "no technique run",
            f"{a.seconds:.1f}s", f"${a.cost_usd:.4f}", "live Claude" if a.live else "offline"]
    if a.cached:
        bits.append(f"**cache hit** ({a.cache_score:.2f})")
    if a.revised:
        bits.append("rewritten by self-check")
    note = ""
    if a.unknown_citations:
        note += f"\n\n:warning: cites chunks it was not given: `{a.unknown_citations}`"
    if a.errors:
        note += f"\n\n:warning: some Claude calls failed and fell back: {a.errors}"
    if a.standins:
        note += "\n\n**Stand-ins in this run:** " + "; ".join(a.standins)
    return " · ".join(bits) + note


# ------------------------------------------------------------------------------------------- the chat

def pairs_from(chat: list[dict]) -> list[tuple[str, str]]:
    """The chatbot keeps a flat list of messages. The agent wants (question, answer) pairs for memory."""
    users = [m["content"] for m in chat if m["role"] == "user"]
    bots = [m["content"] for m in chat if m["role"] == "assistant"]
    return list(zip(users, bots))[-3:]


def run_in_thread(question, history, forced):
    """Run the agent in its own thread and yield its events as they happen."""
    q: queue.Queue = queue.Queue()

    def work():
        try:
            q.put(("done", agent.run(question, get_index(), history, forced, emit=lambda k, p: q.put((k, p)),
                                     cache=get_cache() if os.environ.get("SEMANTIC_CACHE", "1") == "1" else None)))
        except Exception as e:                                           # never leave the screen hanging
            q.put(("error", e))

    threading.Thread(target=work, daemon=True).start()
    while True:
        kind, payload = q.get()
        yield kind, payload
        if kind in ("done", "error"):
            return


def respond(message, chat, technique, used):
    chat = list(chat or [])
    message = (message or "").strip()
    if have_key() and used >= MAX_LIVE_QUESTIONS:
        chat += [{"role": "user", "content": message},
                 {"role": "assistant", "content": f"This demo allows {MAX_LIVE_QUESTIONS} live questions per visit, to keep the bill sane. Reload to start again."}]
        yield chat, "", render_steps([]), "", "", used
        return
    history = pairs_from(chat)
    chat += [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    steps, text = [], ""
    forced = None if technique == AUTO else next((k for k, t in TECHNIQUES.items() if t.label == technique), None)
    yield chat, "", render_steps(steps), "", "", used
    for kind, payload in run_in_thread(message, history, forced):
        if kind == "step":
            steps.append(payload)
        elif kind == "token":
            text += payload
            chat[-1] = {"role": "assistant", "content": text}
        elif kind == "replace":
            text = payload
            chat[-1] = {"role": "assistant", "content": text}
        elif kind == "error":
            chat[-1] = {"role": "assistant", "content": f"Something broke on my side ({type(payload).__name__}). Try again."}
            yield chat, "", render_steps(steps), "", "", used
            return
        elif kind == "done":
            chat[-1] = {"role": "assistant", "content": payload.text}
            used += 0 if payload.status == "blocked" else 1          # a blocked question cost nothing, so it is free
            yield chat, "", render_steps(steps), render_sources(payload.sources), render_footer(payload), used
            return
        yield chat, "", render_steps(steps), "", "", used


# ----------------------------------------------------------------------------------------- compare

def compare(question: str, used: int):
    question = (question or "").strip()
    if not question:
        return [], "Type a question first.", used
    if have_key() and used + 3 > MAX_LIVE_QUESTIONS:
        return [], f"Not enough live questions left this visit (a comparison counts as 3).", used
    ix = get_index()
    rows = []
    with tracing.request_trace("compare", question=question[:200]) as tr:
        for t in TECHNIQUES.values():
            try:
                r = t.fn(ix, question, 3)
                rows.append([t.label, "\n".join(d.meta["chunk_id"] for d in r.documents) or "(nothing)", r.notes[0] if r.notes else ""])
            except Exception as e:
                rows.append([t.label, "(failed)", f"{type(e).__name__}: {e}"])
        cost = tr.cost_usd
    msg = f"Same question through every technique. Cost of this comparison: ${cost:.4f}." + ("" if have_key() else " Offline stand-ins in use; see the About tab.")
    return rows, msg, used + (3 if have_key() else 0)


# --------------------------------------------------------------------------------------------- about

def about_text() -> str:
    embedding_line = (get_index().embedder.status if _index is not None else
                      "a neural model (bge-small through fastembed) if it can be downloaded, otherwise a TF-IDF + SVD stand-in. "
                      "Ask a question and the footer says which one ran.")
    rows = "\n".join(f"| {t.label} | {t.when} |" for t in TECHNIQUES.values())
    return f"""
### What this is
One agent over a small set of invented marketing documents (a snack brand, its campaigns, agencies and a media mix model).
It picks a retrieval technique, checks the evidence, switches technique if the evidence is weak, and answers with citations.

### The techniques
| Technique | Use it when |
|---|---|
{rows}

### What is real and what is a stand-in
- **Always real:** the Haystack pipelines, Qdrant storage and filters, BM25, reciprocal rank fusion, parent-document merging, the summary tree, personalised PageRank, the semantic cache, the citation check.
- **Needs a Claude key:** planning by Claude, grading, HyDE pretend answers, multi-query rewrites, decomposition, contextual lines, summaries, entity extraction, the self-check, written answers.
- **Embeddings:** {embedding_line}
- **Late chunking** uses a toy context encoder unless you install the long-context model (see the README). **ColBERT** uses real token vectors only with a neural embedder.
- The documents are invented. No real company data is in this repo.
"""


# ---------------------------------------------------------------------------------------------- build

def build() -> gr.Blocks:
    with gr.Blocks(title="RAG technique agent") as demo:
        gr.Markdown("# RAG technique agent\nAsk a question about a made-up snack brand's marketing. Watch the agent choose how to search.")
        gr.Markdown(banner())
        used = gr.State(0)
        with gr.Tabs():
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(label="Chat", height=420)
                        msg = gr.Textbox(label="Your question", placeholder="Ask about campaigns, agencies, audiences, the price increase...")
                        tech = gr.Dropdown([AUTO] + [t.label for t in TECHNIQUES.values()], value=AUTO, label="Technique")
                        gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[msg], label="Try one")
                        footer = gr.Markdown()
                    with gr.Column(scale=2):
                        gr.Markdown("### How the agent decided")
                        steps_md = gr.Markdown(render_steps([]))
                        gr.Markdown("### Chunks it used")
                        sources_md = gr.Markdown()
                outs = [chatbot, msg, steps_md, sources_md, footer, used]
                msg.submit(respond, [msg, chatbot, tech, used], outs)
            with gr.Tab("Compare techniques"):
                q2 = gr.Textbox(label="Question", value=EXAMPLES[0])
                go = gr.Button("Run through every technique")
                note = gr.Markdown()
                table = gr.Dataframe(headers=["Technique", "Top 3 chunks", "What it did"], wrap=True, interactive=False)
                go.click(compare, [q2, used], [table, note, used])
            with gr.Tab("About"):
                gr.Markdown(about_text())
    return demo


if __name__ == "__main__":
    build().launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), theme=gr.themes.Soft())
