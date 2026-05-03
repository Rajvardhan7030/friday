import pytest
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from friday.core.agent_runner import AgentRunner
from friday.core.config import Config
from friday.llm.engine import LLMResponse

@pytest.mark.asyncio
async def test_memory_tiering(tmp_path):
    # 1. Setup Config
    persist_dir = tmp_path / "memory"
    persist_dir.mkdir()
    
    config = Config()
    config.set("memory.enabled", True)
    config.set("memory.persist_directory", str(persist_dir))
    config.set("llm.engine", "mock")
    
    # 2. Mock LLM
    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.model_name = "mock-model"
    
    # Mock fact extraction
    fact_extraction_response = LLMResponse(content='["User likes pizza", "User is a coder"]')
    # Mock general chat
    chat_response = LLMResponse(content="Hello! I remember you like pizza.")
    
    # Sequence of responses
    mock_llm.chat = AsyncMock(side_effect=[
        chat_response,           # for handle_input
        fact_extraction_response # for consolidation
    ])
    mock_llm.embed = AsyncMock(return_value=[0.1] * 384)
    mock_llm.embed_batch = AsyncMock(return_value=[[0.1] * 384] * 2)
    mock_llm.aclose = AsyncMock()
    
    # 3. Initialize AgentRunner
    runner = AgentRunner(config)
    runner.llm = mock_llm # Inject mock
    
    # 4. Simulate conversation
    await runner.handle_input("I really love pizza and I work as a coder.")
    
    # Check if messages were added to STM (SQLite)
    assert runner.conversation_memory is not None
    history = await runner.conversation_memory.get_history(runner.session.session_id)
    assert len(history) >= 2
    
    # 5. Trigger Consolidation (via aclose)
    await runner.aclose()
    
    # 6. Verify LTM storage
    # Re-init runner to check retrieval
    runner2 = AgentRunner(config)
    runner2.llm = mock_llm
    
    # Mock retrieval search
    mock_llm.embed.return_value = [0.1] * 384
    
    # Ensure memory is ready
    await runner2._ensure_memory_ready()
    
    # Check LTM collection
    ltm_collection = config.get("memory.ltm_collection", "ltm_memory")
    ltm_results = await runner2.vector_store.similarity_search("pizza", collection_name=ltm_collection)
    assert len(ltm_results) > 0
    assert any("pizza" in r["content"].lower() for r in ltm_results)
    
    print("Memory Tiering Verification Successful!")

if __name__ == "__main__":
    asyncio.run(test_memory_tiering(Path("./tmp"), None))
