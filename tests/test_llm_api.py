"""Tests for OpenAI-compatible API engines."""

from unittest.mock import AsyncMock

import pytest

from friday.llm.api import GeminiEngine, create_api_engine, _is_retryable_api_error
from unittest.mock import AsyncMock, MagicMock

def test_is_retryable_api_error_handles_429():
    import httpx
    
    # 1. Normal 429
    response_normal = MagicMock(status_code=429, text="Too many requests")
    error_normal = httpx.HTTPStatusError("429", request=MagicMock(), response=response_normal)
    assert _is_retryable_api_error(error_normal) is True
    
    # 2. Quota 429
    response_quota = MagicMock(status_code=429, text="Resource has been exhausted (e.g. check quota).")
    error_quota = httpx.HTTPStatusError("429", request=MagicMock(), response=response_quota)
    assert _is_retryable_api_error(error_quota) is False
    
    # 3. 500 Error
    response_500 = MagicMock(status_code=500, text="Internal Server Error")
    error_500 = httpx.HTTPStatusError("500", request=MagicMock(), response=response_500)
    assert _is_retryable_api_error(error_500) is True


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
        {"model": "gemini-embedding-001", "input": ["how to make omlette"]},
    )
