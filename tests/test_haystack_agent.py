from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall

from rag_agent import haystack_agent
from rag_agent.techniques import TECHNIQUES


def scripted(*steps):
    return MockChatGenerator(responses=list(steps))


def test_tools_are_the_techniques_with_their_descriptions(offline_index):
    tools = haystack_agent.make_tools(offline_index)
    names = {t.name for t in tools}
    assert "contextual" in names and "naive" not in names and len(tools) == len(TECHNIQUES) - 1
    assert all(t.description for t in tools)


def test_agent_calls_a_tool_then_answers_and_state_holds_the_chunks(offline_index):
    call = ToolCall(tool_name="contextual", arguments={"question": "How much did Crunch Time spend?"})
    gen = scripted(ChatMessage.from_assistant(tool_calls=[call]),
                   ChatMessage.from_assistant("Crunch Time spent 9.5 million dollars [campaign-crunch-time/results]."))
    out = haystack_agent.ask(offline_index, "How much did Crunch Time spend?", gen)
    assert out["tools_called"] == ["contextual"]
    assert "campaign-crunch-time/results" in out["chunks"]                      # collected in State by the tool
    assert "9.5 million" in out["answer"]


def test_agent_can_try_a_second_tool_when_the_first_is_not_enough(offline_index):
    q = {"question": "lessons across campaigns"}
    gen = scripted(ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="hybrid", arguments=q)]),
                   ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="raptor", arguments=q)]),
                   ChatMessage.from_assistant("Done."))
    out = haystack_agent.ask(offline_index, "lessons across campaigns?", gen)
    assert out["tools_called"] == ["hybrid", "raptor"]


def test_without_a_key_and_without_a_generator_it_says_so(offline_index):
    import pytest
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        haystack_agent.build_agent(offline_index)
