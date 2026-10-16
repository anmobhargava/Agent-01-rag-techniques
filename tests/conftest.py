import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["TRACE_LOG"] = "0"
os.environ["EMBEDDER"] = "lsa"                      # tests never touch the network or download a model
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "False"
os.environ.pop("ANTHROPIC_API_KEY", None)          # tests start offline; a test that needs "live" turns it on with the fake client

import pytest


@pytest.fixture(scope="session")
def offline_index():
    from rag_agent.index import Index
    return Index()


@pytest.fixture
def fake_claude(monkeypatch):
    """Pretend an API key is set and route every Anthropic call to the scripted fake."""
    import anthropic
    from fake_claude import FakeClient
    FakeClient.calls, FakeClient.systems = [], []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-tests")
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    return FakeClient


@pytest.fixture(autouse=True)
def private_cache(tmp_path, monkeypatch):
    """Tests must never write to the real data/cache: a fake model's output would be reused by a real run later."""
    from rag_agent import index
    monkeypatch.setattr(index, "CACHE_DIR", tmp_path / "cache")
