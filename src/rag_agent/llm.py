"""
One place that talks to Claude.

Every model call in this repo goes through ask(). That gives me three things for free:
  1. tracing and cost (each call shows up in the trace with tokens and dollars),
  2. one fallback rule: no API key, or the call fails -> return the caller's `offline` text instead of crashing,
  3. one spot to change a model name.

Why the offline text exists: I built this without an Anthropic key to hand, so every technique that
needs a model also has a plain, rule-based stand-in. The stand-ins keep the demo and the tests running, but
they are NOT the real technique, and the app says so whenever one was used. With a key set, none of them run.
"""

import os

from . import tracing

FAST = os.environ.get("FAST_MODEL", "claude-haiku-4-5-20251001")      # grading, routing, query rewriting
STRONG = os.environ.get("ANSWER_MODEL", "claude-sonnet-4-6")           # the final answer and summaries


def have_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _note(kind: str, text: str) -> None:
    """Remember on the current trace that something fell back, so the UI can say so."""
    tr = tracing.current()
    if tr is not None:
        tr.meta.setdefault(kind, []).append(text)


def ask(system: str, user: str, call_type: str, model: str = FAST, max_tokens: int = 500, offline: str = "") -> str:
    """Ask Claude one thing. Returns `offline` when there is no key or the call fails."""
    if not have_key():
        _note("standins", call_type)
        return offline
    try:
        import anthropic
        client = anthropic.Anthropic()
        with tracing.llm_span(call_type, model) as rec:
            rec["response"] = client.messages.create(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}])
        return rec["response"].content[0].text.strip()
    except Exception as e:                                  # an API hiccup should not take the demo down
        _note("errors", f"{call_type}: {type(e).__name__}")
        return offline


def ask_stream(system: str, user: str, call_type: str, model: str = STRONG, max_tokens: int = 700, offline: str = ""):
    """Same as ask() but yields the answer in pieces, so the chat can show it as it is written."""
    if not have_key():
        _note("standins", call_type)
        for word in offline.split(" "):
            yield word + " "
        return
    started = False
    try:
        import anthropic
        client = anthropic.Anthropic()
        with tracing.llm_span(call_type, model) as rec:
            with client.messages.stream(model=model, max_tokens=max_tokens, system=system,
                                        messages=[{"role": "user", "content": user}]) as stream:
                for piece in stream.text_stream:
                    started = True
                    yield piece
                rec["response"] = stream.get_final_message()
    except Exception as e:
        _note("errors", f"{call_type}: {type(e).__name__}")
        if not started:                       # if some of the answer already went out, do not glue the fallback on after it
            yield offline
