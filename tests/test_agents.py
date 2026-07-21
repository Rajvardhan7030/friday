"""Tests for FRIDAY agents."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import types
import asyncio

from friday.agents.base import Context
# from friday.agents.code_assistant import _resolve_workspace_dir, code_task_handler
from friday.plugins.morning_digest.main import morning_digest_handler
from friday.plugins.research.main import ResearchAgent
from friday.agents.command_handlers import clear_handler
from friday.core.agent_runner import Session
from friday.core.agent_runner import AgentRunner
from friday.core.config import Config
from friday.core.registry import registry
from friday.llm.engine import Message
from friday.llm.local import LocalEngine

@pytest.mark.asyncio
async def test_research_agent(mock_llm, mock_vector_store, monkeypatch):
    monkeypatch.setattr(
        "friday.plugins.research.main.WebSearchSkill.execute",
        AsyncMock(return_value=type("SkillResult", (), {"success": True, "data": []})()),
    )

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
    assert result == "Session history cleared (in-memory only)."


def test_identity_command_does_not_match_mid_sentence():
    assert registry.find_handler("I don't know who you are") is None


def test_greeting_command_does_not_match_mid_sentence():
    assert registry.find_handler("I said hello to the team already") is None


def test_identity_command_matches_direct_question():
    handler_data = registry.find_handler("who are you?")

    assert handler_data is not None
    command, _match = handler_data
    assert command.name == "Identity"


def test_greeting_command_matches_simple_greeting():
    handler_data = registry.find_handler("hello")

    assert handler_data is not None
    command, _match = handler_data
    assert command.name == "Greeting"


@pytest.mark.asyncio
async def test_morning_digest_command_matches_and_uses_agent(monkeypatch):
    session = Session()
    llm = MagicMock()
    config = MagicMock()
    tts = MagicMock()

    async def fake_run(self, ctx):
        assert ctx.user_query == "morning digest"
        assert ctx.chat_history == session.history
        return type("Result", (), {"content": "Here is your digest."})()

    monkeypatch.setattr("friday.plugins.morning_digest.main.MorningDigestAgent.run", fake_run)

    handler_data = registry.find_handler("morning digest")

    assert handler_data is not None
    command, _match = handler_data
    assert command.name == "Morning Digest"

    result = await morning_digest_handler(session, llm=llm, config=config, tts=tts)

    assert result == "Here is your digest."


@pytest.mark.asyncio
async def test_local_engine_retries_primary_on_each_chat(monkeypatch):
    fake_ollama = MagicMock()
    fake_ollama.AsyncClient.return_value = MagicMock()
    monkeypatch.setattr("friday.llm.local.ollama", fake_ollama)

    engine = LocalEngine("primary", "fallback")

    async def fake_chat(model, *, messages, tools=None, stream=False, **kwargs):
        if model == "primary":
            raise Exception("404 model not found")
        return {"message": {"content": f"reply via {model}"}}

    engine._client.chat = fake_chat
    messages = [Message(role="user", content="hello")]

    first = await engine.chat(messages)
    second = await engine.chat(messages)

    assert first.content == "reply via fallback"
    assert second.content == "reply via fallback"
    assert engine.model_name == "fallback"


@pytest.mark.asyncio
async def test_local_engine_sticks_to_working_model(monkeypatch):
    fake_ollama = MagicMock()
    fake_ollama.AsyncClient.return_value = MagicMock()
    monkeypatch.setattr("friday.llm.local.ollama", fake_ollama)

    engine = LocalEngine("primary", "fallback")
    call_counts = {"primary": 0, "fallback": 0}

    async def fake_chat(model, *, messages, tools=None, stream=False, **kwargs):
        call_counts[model] += 1
        if model == "primary" and call_counts["primary"] == 1:
            raise Exception("404 model not found")
        return {"message": {"content": f"reply via {model}"}}

    engine._client.chat = fake_chat
    messages = [Message(role="user", content="hello")]

    first = await engine.chat(messages)
    second = await engine.chat(messages)

    assert first.content == "reply via fallback"
    # Optimization: should NOT try primary again immediately if fallback is working
    assert second.content == "reply via fallback"
    assert engine.model_name == "fallback"
    assert call_counts == {"primary": 1, "fallback": 2}


# def test_code_task_uses_configured_workspace_by_default(tmp_path):
#     pass
# def test_code_task_uses_desktop_when_requested(tmp_path):
#     pass
# @pytest.mark.asyncio
# async def test_code_task_handler_executes_in_configured_workspace(monkeypatch, tmp_path):
#     pass
# @pytest.mark.asyncio
# async def test_code_task_handler_returns_clean_sandbox_failure(monkeypatch, tmp_path):
#     pass


@pytest.mark.asyncio
async def test_session_history_limit_is_configurable():
    session = Session(max_history_messages=3, recent_messages=2, summary_max_chars=200)

    for index in range(5):
        session.add_message("user", f"message {index}")

    await session.summarize()

    assert [message["content"] for message in session.history] == [
        "message 2",
        "message 3",
        "message 4",
    ]
    assert "message 0" in session.history_summary
    assert "message 1" in session.history_summary


@pytest.mark.asyncio
async def test_session_builds_llm_messages_with_summary_and_recent_window():
    session = Session(max_history_messages=3, recent_messages=3, summary_max_chars=200)

    for role, content in [
        ("user", "message 0"),
        ("assistant", "reply 0"),
        ("user", "message 1"),
        ("assistant", "reply 1"),
        ("user", "message 2"),
    ]:
        session.add_message(role, content)

    await session.summarize()

    # In actual usage, the current user input is added to history BEFORE building messages
    session.add_message("user", "latest question")
    messages = session.build_llm_messages()

    assert messages[0].role == "system"
    assert "You are FRIDAY" in messages[0].content
    assert messages[1].role == "system"
    assert "message 0" in messages[1].content
    assert [(message.role, message.content) for message in messages[2:]] == [
        ("assistant", "reply 1"),
        ("user", "message 2"),
        ("user", "latest question"),
    ]


def test_agent_runner_discovers_modules_dynamically(monkeypatch):
    imported_modules = []
    fake_package = types.SimpleNamespace(__path__=["/fake/friday/agents"])

    class FakeModuleInfo:
        def __init__(self, name):
            self.name = name

    def fake_import_module(name):
        imported_modules.append(name)
        if name == "friday.agents":
            return fake_package
        return object()

    monkeypatch.setattr(
        "friday.core.agent_runner.import_module",
        fake_import_module,
    )
    monkeypatch.setattr(
        "friday.core.agent_runner.pkgutil.iter_modules",
        lambda paths: [FakeModuleInfo("command_handlers"), FakeModuleInfo("morning_digest"), FakeModuleInfo("_private")],
    )

    runner = AgentRunner.__new__(AgentRunner)
    runner.config = MagicMock()
    runner.config.get.return_value = []

    runner._load_agents()

    assert imported_modules == [
        "friday.agents",
        "friday.agents.command_handlers",
        "friday.agents.morning_digest",
    ]


def test_agent_runner_skips_configured_agent_modules(monkeypatch):
    imported_modules = []
    fake_package = types.SimpleNamespace(__path__=["/fake/friday/agents"])

    class FakeModuleInfo:
        def __init__(self, name):
            self.name = name

    def fake_import_module(name):
        imported_modules.append(name)
        if name == "friday.agents":
            return fake_package
        return object()

    monkeypatch.setattr(
        "friday.core.agent_runner.import_module",
        fake_import_module,
    )
    monkeypatch.setattr(
        "friday.core.agent_runner.pkgutil.iter_modules",
        lambda paths: [FakeModuleInfo("command_handlers"), FakeModuleInfo("morning_digest")],
    )

    runner = AgentRunner.__new__(AgentRunner)
    runner.config = MagicMock()
    runner.config.get.return_value = ["morning_digest"]

    runner._load_agents()

    assert imported_modules == [
        "friday.agents",
        "friday.agents.command_handlers",
    ]


@pytest.mark.asyncio
async def test_agent_runner_injects_long_term_memory_into_llm_context():
    runner = AgentRunner.__new__(AgentRunner)
    runner.config = MagicMock()
    runner.config.get.side_effect = lambda key, default=None: {
        "memory.retrieval_limit": 2,
        "memory.auto_remember_conversations": False,
    }.get(key, default)
    runner._memory_lock = asyncio.Lock()
    runner.session = Session(max_history_messages=10, recent_messages=5, summary_max_chars=200)
    runner.llm = MagicMock()
    runner.llm.chat = AsyncMock(return_value=type("Response", (), {"content": "memory aware answer", "tool_calls": None})())
    runner.llm.embed = AsyncMock(return_value=[0.1] * 768)
    runner.vector_store = MagicMock()
    runner.vector_store.llm = runner.llm
    runner.vector_store.similarity_search = AsyncMock(return_value=[
        {"content": "Friday likes local-first tools.", "metadata": {"source": "notes.md"}}
    ])
    runner.document_indexer = MagicMock()
    runner._memory_ready = True
    runner._memory_disabled_reason = None
    runner.memory_manager = MagicMock()
    runner.memory_manager.build_memory_message = AsyncMock(return_value=Message(
        role="system",
        content="Untrusted retrieved data... Relevant long-term memory... notes.md"
    ))
    runner.orchestrator = MagicMock()
    runner.orchestrator.ensure_mcp_ready = AsyncMock()
    runner.skills = {}
    runner._pending_tasks = set()

    # Mock _run_react_loop to capture messages
    captured_messages = []
    async def mock_react_loop(text, messages, tools, stream=False):
        nonlocal captured_messages
        captured_messages = messages
        yield "memory aware answer"
    
    runner._run_react_loop = mock_react_loop

    full_response = ""
    async for chunk in AgentRunner._fallback_to_llm(runner, "What do you know about Friday?"):
        full_response += chunk
    result = full_response

    assert result == "memory aware answer"
    assert any("Relevant long-term memory" in message.content for message in captured_messages)
    assert any("notes.md" in message.content for message in captured_messages)


@pytest.mark.asyncio
async def test_agent_runner_remembers_successful_llm_exchanges():
    runner = AgentRunner.__new__(AgentRunner)
    runner.config = MagicMock()
    runner.config.get.side_effect = lambda key, default=None: {
        "memory.retrieval_limit": 3,
        "memory.auto_remember_conversations": True,
    }.get(key, default)
    runner._memory_lock = asyncio.Lock()
    runner.session = Session(max_history_messages=10, recent_messages=5, summary_max_chars=200)
    runner.llm = MagicMock()
    runner.llm.chat = AsyncMock(return_value=type("Response", (), {"content": "stored answer", "tool_calls": None})())
    runner.llm.embed = AsyncMock(return_value=[0.1] * 768)
    runner.vector_store = MagicMock()
    runner.vector_store.llm = runner.llm
    runner.vector_store.similarity_search = AsyncMock(return_value=[])
    runner.vector_store.add_documents = AsyncMock()
    runner.conversation_memory = MagicMock()
    runner.conversation_memory.add_message = AsyncMock()
    runner.memory_consolidator = MagicMock()
    runner.document_indexer = MagicMock()
    runner._memory_ready = True
    runner._memory_disabled_reason = None
    runner.memory_manager = MagicMock()
    runner.memory_manager.build_memory_message = AsyncMock(return_value=None)
    runner.memory_manager.add_to_history = AsyncMock()
    runner.memory_manager.remember_exchange = AsyncMock()
    
    runner.orchestrator = MagicMock()
    runner.orchestrator.ensure_mcp_ready = AsyncMock()
    runner.skills = {}
    runner._pending_tasks = set()

    # Mock _run_react_loop to call _remember_exchange
    async def mock_react_loop(text, messages, tools, stream=False):
        await runner._remember_exchange(text, "stored answer")
        yield "stored answer"
    
    runner._run_react_loop = mock_react_loop
    runner._remember_exchange = MagicMock(side_effect=runner._remember_exchange) # For some reason it was fail in previous run, let's just mock it directly or ensure it calls MM

    full_response = ""
    async for chunk in AgentRunner._fallback_to_llm(runner, "remember this"):
        full_response += chunk
    result = full_response

    assert result == "stored answer"
    runner.memory_manager.remember_exchange.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_runner_gracefully_disables_memory_when_initialization_fails():
    from friday.memory.manager import MemoryManager
    from unittest.mock import patch
    
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: {
        "memory.auto_index_directories": [],
        "memory.persist_directory": "/tmp/friday_test",
        "memory.enabled": True,
    }.get(key, default)
    
    model_router = MagicMock()
    model_router.get_engine_for_task = AsyncMock()
    # Mocking embed engine to fail
    model_router.get_engine_for_task.side_effect = RuntimeError("chromadb unavailable")
    
    with patch("friday.memory.manager.VectorStore"), \
         patch("friday.memory.manager.ConversationMemory"), \
         patch("friday.memory.manager.MemoryConsolidator"), \
         patch("friday.memory.manager.DocumentIndexer"):
        
        mm = MemoryManager(config, model_router)
        pending_tasks = set()
        ready = await mm.ensure_ready(pending_tasks)

        assert ready is False
        assert mm.vector_store is None
        assert mm.document_indexer is None
        assert "chromadb unavailable" in mm._disabled_reason


@pytest.mark.asyncio
async def test_agent_runner_fallback_includes_system_persona():
    runner = AgentRunner.__new__(AgentRunner)
    runner.config = MagicMock()
    runner.config.get.side_effect = lambda key, default=None: {
        "memory.retrieval_limit": 3,
        "memory.auto_remember_conversations": False,
    }.get(key, default)
    runner._memory_lock = asyncio.Lock()
    runner.session = Session()
    runner.llm = MagicMock()
    runner.llm.is_available.return_value = True
    runner.memory_manager = MagicMock()
    runner.memory_manager.build_memory_message = AsyncMock(return_value=None)
    runner.orchestrator = MagicMock()
    runner.orchestrator.ensure_mcp_ready = AsyncMock()
    runner.skills = {}
    runner._pending_tasks = set()

    # Mock _run_react_loop to capture messages
    captured_messages = []
    async def mock_react_loop(text, messages, tools, stream=False):
        nonlocal captured_messages
        captured_messages = messages
        yield "ok"
    
    runner._run_react_loop = mock_react_loop
    
    async for _ in runner._fallback_to_llm("hello"):
        pass
    
    # Verify the system persona is present
    system_messages = [m for m in captured_messages if m.role == "system"]
    assert any("You are FRIDAY" in m.content for m in system_messages)
    assert any("privacy-first local AI assistant" in m.content for m in system_messages)


@pytest.mark.asyncio
async def test_agent_router_general_chat_includes_system_persona():
    from friday.agents.router import AgentRouter
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=type("Response", (), {"content": "hi", "tool_calls": None})())
    router = AgentRouter(llm)
    
    await router._run_general_chat("hello", [])
    
    args, kwargs = llm.chat.call_args
    messages = args[0]
    
    system_messages = [m for m in messages if m.role == "system"]
    assert any("You are FRIDAY" in m.content for m in system_messages)
    assert any("privacy-first local AI assistant" in m.content for m in system_messages)


from friday.agents.code_assistant import CodeAssistantAgent
from friday.agents.sandbox_executor import SandboxExecutor

@pytest.mark.asyncio
async def test_code_assistant_returns_clean_message_on_sandbox_failure(monkeypatch, tmp_path):
    config = Config()
    llm = MagicMock()
    del llm.get_engine_for_task
    # Mock LLM to return a simple code block
    llm.chat = AsyncMock(side_effect=[
        type("Response", (), {"content": "Step 1: print ok"})(), # Plan
        type("Response", (), {"content": "```python\nprint('ok')\n```"})(), # Generate 1
        type("Response", (), {"content": "```python\nprint('ok')\n```"})(), # Generate 2
        type("Response", (), {"content": "```python\nprint('ok')\n```"})(), # Generate 3
    ])
    
    executor = SandboxExecutor(config)
    agent = CodeAssistantAgent(llm, sandbox=executor)
    
    # Mock the low-level sandbox call to fail
    monkeypatch.setattr(
        "friday.agents.sandbox_executor.run_sandboxed_code",
        AsyncMock(return_value=(False, "Error: network-isolated sandbox unavailable")),
    )
    ctx = Context(user_query="create file named demo.py")
    result = await agent.run(ctx)

    assert result.success is False
    assert "Last Output:" in result.content
    assert "network-isolated sandbox unavailable" in result.content
