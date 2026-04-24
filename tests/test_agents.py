"""Tests for FRIDAY agents."""

import pytest

from friday.agents.base import Context
from friday.agents.research import ResearchAgent
from friday.agents.system_commands import clear_handler
from friday.core.agent_runner import Session
from friday.core.registry import registry

@pytest.mark.asyncio
async def test_research_agent(mock_llm, mock_vector_store):
    agent = ResearchAgent(mock_llm, mock_vector_store)
    ctx = Context(user_query="Who is Friday?")
    
    result = await agent.run(ctx)
    
    assert result.success is True
    assert "Mocked response" in result.content
    assert len(result.citations) > 0
    assert result.citations[0]["source"] == "test.txt"


@pytest.mark.asyncio
async def test_clear_command_matches_plain_clear():
    handler_data = registry.find_handler("clear")

    assert handler_data is not None

    command, _match = handler_data
    session = Session()
    session.history = [{"role": "user", "content": "hello"}]

    result = await clear_handler(session)

    assert command.name == "Clear"
    assert session.history == []
    assert result == "Conversation history cleared."
