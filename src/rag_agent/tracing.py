"""
Tracing and cost: "what happened inside that one request, and what did it cost?"

Every question gets a Trace. Inside it I record two kinds of thing:
  spans      the steps (find seeds, rank, gather facts), each with how long it took
  llm calls  every model call: which kind (answer, extract), which model, tokens in/out, dollars

At the end of the request the trace is printed as ONE JSON line to stdout, so whatever collects logs
(Docker, a Hugging Face Space, CloudWatch) can search it. If Langfuse keys are set, the same trace is also
sent to Langfuse, where I get the dashboard (cost by call type, slowest steps, one trace per question).

The rule I care most about: tracing must never break the answer. Every Langfuse call is wrapped so
that if it throws, the request carries on untraced.

What gets recorded: the question (first 200 characters) and the step timings. This demo runs on invented
data, so the question is safe to keep. For real customer data I would log the length and a hash, not the text.

Prices are an estimate, kept in one table. Check the current pricing page before quoting a dollar figure.
I have not run this against a live Langfuse yet (no keys). The log line and the cost maths are tested.
"""

import contextvars
import json
import os
import time
import uuid
from contextlib import ExitStack, contextmanager

# USD per million tokens: (input, output). Unknown model: assume the dearer price, never under-report.
PRICES = {"claude-haiku-4-5-20251001": (1.0, 5.0), "claude-sonnet-4-6": (3.0, 15.0), "claude-sonnet-5-5": (3.0, 15.0)}
_DEFAULT_PRICE = (3.0, 15.0)

_current: contextvars.ContextVar = contextvars.ContextVar("trace", default=None)


def _price(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = PRICES.get(model, _DEFAULT_PRICE)
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


class Trace:
    def __init__(self, name: str, meta: dict):
        self.id, self.name, self.meta = uuid.uuid4().hex[:16], name, meta
        self.t0 = time.time()
        self.spans: list[dict] = []
        self.llm: list[dict] = []

    @property
    def cost_usd(self) -> float:
        return round(sum(c["cost_usd"] for c in self.llm), 6)

    def summary(self) -> dict:
        return {"trace_id": self.id, "name": self.name, **self.meta,
                "seconds": round(time.time() - self.t0, 3),
                "spans": self.spans, "llm_calls": self.llm,
                "tokens_in": sum(c["tokens_in"] for c in self.llm),
                "tokens_out": sum(c["tokens_out"] for c in self.llm),
                "cost_usd": self.cost_usd}


def current() -> "Trace | None":
    return _current.get()


# ----------------------------------------------------------------------------------- langfuse (optional)

def _langfuse_on() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def _enter_langfuse(stack: ExitStack, name: str, as_type: str = "span", **kw):
    """Open a Langfuse observation on `stack`. Returns it, or None if Langfuse is off or throws.
    The try/except covers ONLY this setup. The caller's own code runs outside it, so a real error in
    the request is never swallowed or disguised as a tracing error."""
    if not _langfuse_on():
        return None
    try:
        from langfuse import get_client
        return stack.enter_context(get_client().start_as_current_observation(name=name, as_type=as_type, **kw))
    except Exception:
        return None


@contextmanager
def request_trace(name: str, **meta):
    """Wrap ONE request. Everything inside (spans, llm calls) belongs to this trace."""
    tr = Trace(name, meta)
    token = _current.set(tr)
    stack = ExitStack()
    lf = _enter_langfuse(stack, name, input=meta.get("question"), metadata={k: v for k, v in meta.items() if k != "question"})
    try:
        yield tr
    finally:
        _current.reset(token)
        try:
            if lf is not None:
                lf.update(output={"cost_usd": tr.cost_usd, "seconds": round(time.time() - tr.t0, 3)})
        except Exception:
            pass
        try:
            stack.close()
            if _langfuse_on():
                from langfuse import get_client
                get_client().flush()          # a serverless host can freeze right after the response, so send now
        except Exception:
            pass
        if os.environ.get("TRACE_LOG", "1") != "0":          # the eval turns this off so it does not print a line per question
            print(json.dumps({"event": "trace", **tr.summary()}), flush=True)


@contextmanager
def span(name: str, **info):
    """Time one step of the pipeline. Does nothing if there is no active trace."""
    tr = current()
    if tr is None:
        yield
        return
    stack = ExitStack()
    _enter_langfuse(stack, name)
    t = time.time()
    try:
        yield
    finally:
        tr.spans.append({"name": name, "ms": round((time.time() - t) * 1000, 1), **info})
        try:
            stack.close()
        except Exception:
            pass


@contextmanager
def llm_span(call_type: str, model: str):
    """Wrap one model call. The caller puts the response in rec["response"]; I read the token counts from it.

    call_type is the label I slice cost by later: "answer", "extract", "report". It is the difference between
    "this agent costs $0.01 a question" and "extraction is 90% of the bill"."""
    rec: dict = {}
    tr = current()
    stack = ExitStack()
    lf = _enter_langfuse(stack, call_type, as_type="generation", model=model)
    t = time.time()
    try:
        yield rec
    finally:
        seconds = round(time.time() - t, 3)
        tokens_in = tokens_out = 0
        try:
            usage = rec["response"].usage
            tokens_in, tokens_out = int(usage.input_tokens or 0), int(usage.output_tokens or 0)
        except Exception:
            pass                                    # no usage on the response: count it as zero rather than crash
        cost = _price(model, tokens_in, tokens_out)
        if tr is not None:
            tr.llm.append({"type": call_type, "model": model, "tokens_in": tokens_in, "tokens_out": tokens_out,
                           "seconds": seconds, "cost_usd": round(cost, 6)})
        try:
            if lf is not None:
                lf.update(usage_details={"input": tokens_in, "output": tokens_out})
            stack.close()
        except Exception:
            pass
