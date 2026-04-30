"""Tests for FRIDAY agents."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import types

from friday.agents.base import Context
# from friday.agents.code_assistant import _resolve_workspace_dir, code_task_handler
from friday.plugins.morning_digest.main import morning_digest_handler
from friday.plugins.research.main import ResearchAgent
from friday.agents.system_commands import clear_handler
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
    assert result == "Conversation history cleared."


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

    async def fake_chat(*, model, messages, tools, stream):
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
async def test_local_engine_uses_primary_again_when_it_becomes_available(monkeypatch):
    fake_ollama = MagicMock()
    fake_ollama.AsyncClient.return_value = MagicMock()
    monkeypatch.setattr("friday.llm.local.ollama", fake_ollama)

    engine = LocalEngine("primary", "fallback")
    call_counts = {"primary": 0, "fallback": 0}

    async def fake_chat(*, model, messages, tools, stream):
        call_counts[model] += 1
        if model == "primary" and call_counts["primary"] == 1:
            raise Exception("404 model not found")
        return {"message": {"content": f"reply via {model}"}}

    engine._client.chat = fake_chat
    messages = [Message(role="user", content="hello")]

    first = await engine.chat(messages)
    second = await engine.chat(messages)

    assert first.content == "reply via fallback"
    assert second.content == "reply via primary"
    assert engine.model_name == "primary"
    assert call_counts == {"primary": 2, "fallback": 1}


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


def test_session_history_limit_is_configurable():
    session = Session(max_history_messages=3, recent_messages=2, summary_max_chars=200)

    for index in range(5):
        session.add_message("user", f"message {index}")

    assert [message["content"] for message in session.history] == [
        "message 2",
        "message 3",
        "message 4",
    ]
    assert "message 0" in session.history_summary
    assert "message 1" in session.history_summary


def test_session_builds_llm_messages_with_summary_and_recent_window():
    session = Session(max_history_messages=3, recent_messages=2, summary_max_chars=200)

    for role, content in [
        ("user", "message 0"),
        ("assistant", "reply 0"),
        ("user", "message 1"),
        ("assistant", "reply 1"),
        ("user", "message 2"),
    ]:
        session.add_message(role, content)

    messages = session.build_llm_messages("latest question")

    assert messages[0].role == "system"
    assert "message 0" in messages[0].content
    assert [(message.role, message.content) for message in messages[1:]] == [
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
        lambda paths: [FakeModuleInfo("system_commands"), FakeModuleInfo("morning_digest"), FakeModuleInfo("_private")],
    )

    runner = AgentRunner.__new__(AgentRunner)
    runner.config = MagicMock()
    runner.config.get.return_value = []

    runner._load_agents()

    assert imported_modules == [
        "friday.agents",
        "friday.agents.system_commands",
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
        lambda paths: [FakeModuleInfo("system_commands"), FakeModuleInfo("morning_digest")],
    )

    runner = AgentRunner.__new__(AgentRunner)
    runner.config = MagicMock()
    runner.config.get.return_value = ["morning_digest"]

    runner._load_agents()

    assert imported_modules == [
        "friday.agents",
        "friday.agents.system_commands",
    ]


@pytest.mark.asyncio
async def test_agent_runner_injects_long_term_memory_into_llm_context():
    runner = AgentRunner.__new__(AgentRunner)
    runner.config = MagicMock()
    runner.config.get.side_effect = lambda key, default=None: {
        "memory.retrieval_limit": 2,
        "memory.auto_remember_conversations": False,
    }.get(key, default)
    runner.session = Session(max_history_messages=10, recent_messages=5, summary_max_chars=200)
    runner.llm = MagicMock()
    runner.llm.chat = AsyncMock(return_value=type("Response", (), {"content": "memory aware answer", "tool_calls": None})())
    runner.vector_store = MagicMock()
    runner.vector_store.similarity_search = AsyncMock(return_value=[
        {"content": "Friday likes local-first tools.", "metadata": {"source": "notes.md"}}
    ])
    runner.document_indexer = MagicMock()
    runner._memory_ready = True
    runner._memory_disabled_reason = None

    result = await AgentRunner._fallback_to_llm(runner, "What do you know about Friday?")

    assert result == "memory aware answer"
    messages = runner.llm.chat.await_args.args[0]
    assert any("Relevant long-term memory" in message.content for message in messages)
    assert any("notes.md" in message.content for message in messages)


@pytest.mark.asyncio
async def test_agent_runner_remembers_successful_llm_exchanges():
    runner = AgentRunner.__new__(AgentRunner)
    runner.config = MagicMock()
    runner.config.get.side_effect = lambda key, default=None: {
        "memory.retrieval_limit": 3,
        "memory.auto_remember_conversations": True,
    }.get(key, default)
    runner.session = Session(max_history_messages=10, recent_messages=5, summary_max_chars=200)
    runner.llm = MagicMock()
    runner.llm.chat = AsyncMock(return_value=type("Response", (), {"content": "stored answer", "tool_calls": None})())
    runner.vector_store = MagicMock()
    runner.vector_store.similarity_search = AsyncMock(return_value=[])
    runner.vector_store.add_documents = AsyncMock()
    runner.document_indexer = MagicMock()
    runner._memory_ready = True
    runner._memory_disabled_reason = None

    result = await AgentRunner._fallback_to_llm(runner, "remember this")

    assert result == "stored answer"
    runner.vector_store.add_documents.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_runner_gracefully_disables_memory_when_initialization_fails():
    runner = AgentRunner.__new__(AgentRunner)
    runner.config = MagicMock()
    runner.config.get.side_effect = lambda key, default=None: {
        "memory.auto_index_directories": [],
    }.get(key, default)
    runner.vector_store = MagicMock()
    runner.vector_store.initialize = AsyncMock(side_effect=RuntimeError("chromadb unavailable"))
    runner.document_indexer = MagicMock()
    runner._memory_ready = False
    runner._memory_disabled_reason = None

    await runner._ensure_memory_ready()

    assert runner.vector_store is None
    assert runner.document_indexer is None
    assert runner._memory_disabled_reason == "chromadb unavailable"


from friday.agents.code_assistant import CodeAssistantAgent
from friday.agents.sandbox_executor import SandboxExecutor

@pytest.mark.asyncio
async def test_code_assistant_returns_clean_message_on_sandbox_failure(monkeypatch, tmp_path):
    config = Config()
    llm = MagicMock()
    # Mock LLM to return a simple code block
    llm.chat = AsyncMock(side_effect=[
        type("Response", (), {"content": "Step 1: print ok"})(), # Plan
        type("Response", (), {"content": "```python\nprint('ok')\n```"})(), # Generate 1
        type("Response", (), {"content": "```python\nprint('ok')\n```"})(), # Generate 2
        type("Response", (), {"content": "```python\nprint('ok')\n```"})(), # Generate 3
    ])
    
    executor = SandboxExecutor(config)
    agent = CodeAssistantAgent(llm, executor=executor)
    
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
