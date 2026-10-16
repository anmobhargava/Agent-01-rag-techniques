"""
A scripted stand-in for the Anthropic client, so I can test the code paths that only run with a real key.

It does NOT test whether Claude is good at these jobs. It tests that my code asks the right way, parses
the replies, and falls back when a reply is garbage. It decides what to reply by looking at the system prompt.
"""
import json
from types import SimpleNamespace


def _usage():
    return SimpleNamespace(input_tokens=100, output_tokens=20)


OVERRIDES: dict = {}          # a test can set OVERRIDES["critique"] = "..." to script one kind of reply


def reply_for(system: str, user: str) -> str:
    for needle, text in OVERRIDES.items():
        if needle in system:
            return text
    if "decide how much retrieval" in system:
        return json.dumps({"level": "single", "why": "One search can answer it."})
    if "check an answer against the context" in system:
        return json.dumps({"supported": True, "unsupported": [], "useful": True})
    if "Your previous answer contained claims" in system:
        return "Leaf Local spent 6.2 million dollars [campaign-leaf-local/results]."
    if "turn a question into metadata filters" in system:
        return json.dumps({"doc_type": "research", "period": None})
    if "several searches" in system:
        return "NONE"
    if "write search questions for training" in system:
        return json.dumps(["leaf local budget", "How much money went into the Leaf Local push?", "what did the creators campaign cost"])
    if "rank text chunks" in system:
        return "[1,0,2]"
    if "Write a short passage" in system:
        return "Leaf Local cost about 6.2 million dollars and returned 2.6 dollars per dollar."
    if "Rewrite the question 3 different ways" in system:
        return json.dumps(["What was the budget of Leaf Local?", "Leaf Local total spend", "cost of the Leaf Local campaign"])
    if "Split the question" in system:
        return json.dumps(["Which agency ran Leaf Local?", "What is Tidewater Creative's fee?"])
    if "check retrieved text chunks" in system:
        return "[0,1]"
    if "choose the best retrieval technique" in system:
        return json.dumps({"technique": "contextual", "why": "Asks for a number about a named campaign."})
    if "rewrite follow-up" in system:
        return "Which agency ran Leaf Local and what is Tidewater Creative's score?"
    if "search context for document chunks" in system:
        return "This chunk is from the Leaf Local campaign report."
    if "summarise marketing report" in system:
        return "Creator-led campaigns returned the most; television faded fast."
    if "list the named things" in system:
        return json.dumps(["Leaf Local", "Tidewater Creative", "TikTok"])
    if "theme reports" in system:
        return "Creators on TikTok drove Leaf Local, run by Tidewater Creative."
    if "Rewrite the question as a short keyword" in system:
        return "leaf local spend"
    if "You answer questions about Crunchwell" in system:
        return "Leaf Local spent 6.2 million dollars [campaign-leaf-local/results]. Also see [made-up/doc]."
    return ""


class FakeClient:
    calls: list = []              # first 40 characters of each system prompt
    systems: list = []            # the full system prompts

    def __init__(self, *a, **k):
        self.messages = SimpleNamespace(create=self._create, stream=self._stream)

    def _create(self, model, max_tokens, system, messages):
        FakeClient.calls.append(system[:40]); FakeClient.systems.append(system)
        text = reply_for(system, messages[0]["content"])
        return SimpleNamespace(content=[SimpleNamespace(text=text)], usage=_usage())

    def _stream(self, model, max_tokens, system, messages):
        FakeClient.calls.append(system[:40]); FakeClient.systems.append(system)
        text = reply_for(system, messages[0]["content"])

        class Ctx:
            text_stream = iter(text.split(" ") and [w + " " for w in text.split(" ")])

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

            def get_final_message(self_inner):
                return SimpleNamespace(usage=_usage())
        return Ctx()
