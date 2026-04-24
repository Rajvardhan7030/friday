"""Tests for FRIDAY agents."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from friday.agents.base import Context
from friday.agents.code_assistant import _resolve_workspace_dir, code_task_handler
from friday.agents.morning_digest import morning_digest_handler
from friday.agents.research import ResearchAgent
from friday.agents.system_commands import clear_handler
from friday.core.agent_runner import Session
from friday.core.registry import registry
from friday.llm.engine import Message
from friday.llm.local import LocalEngine

@pytest.mark.asyncio
async def test_research_agent(mock_llm, mock_vector_store, monkeypatch):
    monkeypatch.setattr(
        "friday.agents.research.WebSearchSkill.execute",
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

    async def fake_run(self, ctx):
        assert ctx.user_query == "morning digest"
        assert ctx.chat_history == session.history
        return type("Result", (), {"content": "Here is your digest."})()

    monkeypatch.setattr("friday.agents.morning_digest.TTSEngine", MagicMock())
    monkeypatch.setattr("friday.agents.morning_digest.MorningDigestAgent.run", fake_run)

    handler_data = registry.find_handler("morning digest")

    assert handler_data is not None
    command, _match = handler_data
    assert command.name == "Morning Digest"

    result = await morning_digest_handler(session, llm=llm, config=config)

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


def test_code_task_uses_configured_workspace_by_default(tmp_path):
    session = Session()
    config = MagicMock()
    config.base_dir = tmp_path / "friday-home"

    workspace_dir = _resolve_workspace_dir(session, "create a file named demo.py", config)

    assert workspace_dir == config.base_dir / "workspace"


def test_code_task_uses_desktop_when_requested(tmp_path):
    session = Session()
    config = MagicMock()
    config.base_dir = tmp_path / "friday-home"

    workspace_dir = _resolve_workspace_dir(session, "create a file on desktop named demo.py", config)

    assert workspace_dir == Path.home() / "Desktop"


@pytest.mark.asyncio
async def test_code_task_handler_executes_in_configured_workspace(monkeypatch, tmp_path):
    session = Session()
    config = MagicMock()
    config.base_dir = tmp_path / "friday-home"

    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=[
        type("Response", (), {"content": "```python\nprint('ok')\n```"})(),
        type("Response", (), {"content": "Completed successfully."})(),
    ])

    recorded_workspace = {}

    def fake_run_sandboxed_code(code, workspace_dir, config=None):
        recorded_workspace["path"] = workspace_dir
        return True, "ok"

    monkeypatch.setattr("friday.agents.code_assistant.run_sandboxed_code", fake_run_sandboxed_code)

    result = await code_task_handler(session, "named demo.py", llm=llm, config=config)

    assert result == "Completed successfully."
    assert recorded_workspace["path"] == config.base_dir / "workspace"


def test_session_history_limit_is_configurable():
    session = Session(max_history_messages=3)

    for index in range(5):
        session.add_message("user", f"message {index}")

    assert [message["content"] for message in session.history] == [
        "message 2",
        "message 3",
        "message 4",
    ]
