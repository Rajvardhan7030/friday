"""Tests for OpenAI-compatible API engines."""

from unittest.mock import AsyncMock

import pytest

from friday.llm.api import GeminiEngine, create_api_engine


def test_create_api_engine_normalizes_stale_gemini_alias():
    engine = create_api_engine(
        "gemini-1.5-flash",
        "test-key",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-embedding-001",
    )

    assert isinstance(engine, GeminiEngine)
    assert engine.model_name == "gemini-2.5-flash"
    assert engine._get_auth_config() == ({"Authorization": "Bearer test-key"}, {})


@pytest.mark.asyncio
async def test_gemini_embed_uses_string_input_payload():
    engine = GeminiEngine(
        "gemini-2.5-flash",
        "test-key",
        embedding_model_name="gemini-embedding-001",
    )
    engine._request = AsyncMock(return_value={"data": [{"embedding": [1.0, 2.0]}]})

    result = await engine.embed("how to make omlette")

    assert result == [1.0, 2.0]
    engine._request.assert_awaited_once_with(
        "embeddings",
        {"model": "gemini-embedding-001", "input": "how to make omlette"},
    )
