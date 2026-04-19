"""Tests for FRIDAY agents."""

import pytest
from src.friday.agents.research import ResearchAgent
from src.friday.agents.base import Context

@pytest.mark.asyncio
async def test_research_agent(mock_llm, mock_vector_store):
    agent = ResearchAgent(mock_llm, mock_vector_store)
    ctx = Context(user_query="Who is Friday?")
    
    result = await agent.run(ctx)
    
    assert result.success is True
    assert "Mocked response" in result.content
    assert len(result.citations) > 0
    assert result.citations[0]["source"] == "test.txt"
