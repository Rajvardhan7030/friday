import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from friday.llm.api import GeminiEngine
from friday.llm.engine import Message

@pytest.mark.asyncio
async def test_gemini_payload_compliance():
    # Mock httpx response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Final answer", "role": "assistant"}}],
        "usage": {}
    }
    
    # Mock AsyncClient
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    
    engine = GeminiEngine(model_name="gemini-1.5-flash", api_key="test-key")
    engine._client = mock_client
    
    # Sequence: User -> Assistant (Tool Call) -> Tool Response
    messages = [
        Message(role="user", content="What is the weather?"),
        Message(
            role="assistant", 
            content="", # Empty content
            tool_calls=[{
                "id": "call_123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"location\": \"London\"}"}
            }]
        ),
        Message(
            role="tool", 
            content="Rainy", 
            tool_call_id="call_123"
        )
    ]
    
    # Trigger request
    await engine.chat(messages)
    
    # Inspect the payload sent to post
    args, kwargs = mock_client.post.call_args
    payload = kwargs["json"]
    
    # Verification 1: Assistant message with tool_calls should NOT have content if it was ""
    assistant_msg = payload["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert "content" not in assistant_msg or assistant_msg["content"] is None
    assert assistant_msg["tool_calls"][0]["id"] == "call_123"
    
    # Verification 2: Tool message should have tool_call_id
    tool_msg = payload["messages"][2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_123"
    
    # Verification 3: Unsupported params should be filtered
    assert "parallel_tool_calls" not in payload
    assert "presence_penalty" not in payload

@pytest.mark.asyncio
async def test_gemini_unsupported_params_filtering():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
    
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    
    engine = GeminiEngine(model_name="gemini-1.5-flash", api_key="test-key")
    engine._client = mock_client
    
    # Payload with unsupported keys
    messages = [Message(role="user", content="hi")]
    # Note: We can't easily pass these through .chat() since it doesn't take **kwargs,
    # but _request is where they are filtered.
    
    payload = {
        "model": "gemini-1.5-flash",
        "messages": [{"role": "user", "content": "hi"}],
        "parallel_tool_calls": True,
        "presence_penalty": 0.5,
        "seed": 1234
    }
    
    await engine._request("chat/completions", payload)
    
    args, kwargs = mock_client.post.call_args
    sent_payload = kwargs["json"]
    
    assert "parallel_tool_calls" not in sent_payload
    assert "presence_penalty" not in sent_payload
    assert "seed" not in sent_payload

from friday.core.exceptions import LLMError

@pytest.mark.asyncio
async def test_gemini_missing_tool_call_id_raises_error():
    engine = GeminiEngine(model_name="gemini-1.5-flash", api_key="test-key")
    
    # Message with missing tool_call_id
    messages = [
        Message(role="user", content="What is the weather?"),
        Message(
            role="assistant", 
            content="",
            tool_calls=[{"id": "call_123", "type": "function", "function": {"name": "test", "arguments": "{}"}}]
        ),
        Message(
            role="tool", 
            content="Rainy"
            # tool_call_id is missing!
        )
    ]
    
    with pytest.raises(LLMError, match="Tool message is missing 'tool_call_id'"):
        await engine.chat(messages)

@pytest.mark.asyncio
async def test_gemini_incoming_multiple_tool_calls():
    # Mock httpx response with multiple tool calls
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "tool_a", "arguments": "{}"}
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "tool_b", "arguments": "{\"x\": 1}"}
                    }
                ]
            }
        }],
        "usage": {"total_tokens": 100}
    }
    
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    
    engine = GeminiEngine(model_name="gemini-1.5-flash", api_key="test-key")
    engine._client = mock_client
    
    messages = [Message(role="user", content="Call two tools")]
    response = await engine.chat(messages)
    
    assert response.content == ""
    assert len(response.tool_calls) == 2
    assert response.tool_calls[0]["function"]["name"] == "tool_a"
    assert response.tool_calls[1]["function"]["name"] == "tool_b"
    assert json.loads(response.tool_calls[1]["function"]["arguments"])["x"] == 1
    assert response.usage["total_tokens"] == 100
