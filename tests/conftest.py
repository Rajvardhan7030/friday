"""Pytest configuration and mocks."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from src.friday.llm.engine import LLMEngine, LLMResponse

@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMEngine)
    llm.chat = AsyncMock(return_value=LLMResponse(content="Mocked response"))
    llm.embed = AsyncMock(return_value=[0.1] * 768)
    llm.model_name = "mock-model"
    return llm

@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.similarity_search = AsyncMock(return_value=[
        {"content": "Local knowledge", "metadata": {"source": "test.txt"}}
    ])
    return store
