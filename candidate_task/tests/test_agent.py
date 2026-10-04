from types import SimpleNamespace

from candidate_rag.agent import Agent
from candidate_rag.core import Passage


def test_only_selected_evidence_is_cited():
    agent = Agent.__new__(Agent)
    agent.planner = lambda **kwargs: SimpleNamespace(search_query="library")
    agent.keyword = SimpleNamespace(search=lambda *args: [(Passage("guide.pdf", 3, 1, "Library hours"), 2.0)])
    agent.vector = SimpleNamespace(search=lambda *args: [])
    agent.synthesizer = lambda **kwargs: SimpleNamespace(answer="The library is open.", source_ids="1, 99")
    result = agent.ask("Library hours?")
    assert result.sources == [{"document": "guide.pdf", "page": 3, "chunk": 1}]
    assert "guide.pdf p.3" in result.text
