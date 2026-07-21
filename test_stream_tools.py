import asyncio
from typing import List, Optional, AsyncIterator

class DummyResponse:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

async def mock_stream() -> AsyncIterator[DummyResponse]:
    yield DummyResponse(content="")
    yield DummyResponse(content="", tool_calls=[{"name": "news"}])

async def test():
    response = mock_stream()
    full_content = ""
    tool_calls = None
    
    async for chunk in response:
        if chunk.content:
            full_content += chunk.content
            print("yield", chunk.content)
        if chunk.tool_calls:
            tool_calls = chunk.tool_calls
            
    print("finished stream loop")
    print("tool_calls:", tool_calls)

asyncio.run(test())
