"""Tests for the SystemCommandAgent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from friday.agents.system_command_agent import SystemCommandAgent
from friday.agents.base import Context
from friday.core.config import Config
from friday.llm.engine import Message

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.chat = AsyncMock()
    return llm

@pytest.fixture
def config(tmp_path):
    # Use a temporary directory for base_dir to avoid touching real files
    c = Config()
    c.base_dir = tmp_path
    c.set("security.shell_command_timeout", 30)
    c.set("security.shell_command_allow_sudo", False)
    return c

@pytest.mark.asyncio
async def test_system_command_agent_extraction_and_execution(mock_llm, config):
    """Test that the agent correctly extracts and executes a command."""
    mock_llm.chat.return_value = MagicMock(content="ls -la")
    agent = SystemCommandAgent(mock_llm, config)
    
    # Mock confirmation and execution
    with patch("friday.agents.system_command_agent.Confirm.ask", return_value=True), \
         patch("friday.agents.system_command_agent.run_shell_command", AsyncMock(return_value=(0, "file1\nfile2", ""))) as mock_run:
        
        ctx = Context(user_query="list all files")
        result = await agent.run(ctx)
        
        assert result.success is True
        assert "ls -la" in result.content
        assert "file1" in result.content
        mock_run.assert_awaited_once()

@pytest.mark.asyncio
async def test_system_command_agent_safety_block(mock_llm, config):
    """Test that dangerous commands are blocked."""
    mock_llm.chat.return_value = MagicMock(content="rm -rf /")
    agent = SystemCommandAgent(mock_llm, config)
    
    ctx = Context(user_query="delete everything")
    result = await agent.run(ctx)
    
    assert result.success is False
    assert "Safety Block" in result.content

@pytest.mark.asyncio
async def test_system_command_agent_user_denial(mock_llm, config):
    """Test that the agent respects user cancellation."""
    mock_llm.chat.return_value = MagicMock(content="ls")
    agent = SystemCommandAgent(mock_llm, config)
    
    with patch("friday.agents.system_command_agent.Confirm.ask", return_value=False):
        ctx = Context(user_query="list files")
        result = await agent.run(ctx)
        
        assert "cancelled by user" in result.content

@pytest.mark.asyncio
async def test_system_command_agent_timeout(mock_llm, config):
    """Test handling of command timeouts."""
    mock_llm.chat.return_value = MagicMock(content="sleep 100")
    agent = SystemCommandAgent(mock_llm, config)
    
    with patch("friday.agents.system_command_agent.Confirm.ask", return_value=True), \
         patch("friday.agents.system_command_agent.run_shell_command", AsyncMock(return_value=(-1, "", "Command timed out"))) as mock_run:
        
        ctx = Context(user_query="sleep for a while")
        result = await agent.run(ctx)
        
        assert result.success is False
        assert "timed out" in result.content

@pytest.mark.asyncio
async def test_system_command_agent_system_dir_protection(mock_llm, config):
    """Test protection of system directories."""
    mock_llm.chat.return_value = MagicMock(content="rm /etc/passwd")
    agent = SystemCommandAgent(mock_llm, config)
    
    ctx = Context(user_query="delete passwd file")
    result = await agent.run(ctx)
    
    assert result.success is False
    assert "Modification of system directory /etc is prohibited" in result.content

@pytest.mark.asyncio
async def test_execute_command_direct(mock_llm, config):
    """Test direct execution without LLM extraction (used by registry commands)."""
    agent = SystemCommandAgent(mock_llm, config)
    
    with patch("friday.agents.system_command_agent.Confirm.ask", return_value=True), \
         patch("friday.agents.system_command_agent.run_shell_command", AsyncMock(return_value=(0, "output", ""))):
        
        result = await agent.execute_command("echo hello")
        assert result.success is True
        assert "echo hello" in result.content
        assert "output" in result.content
        # LLM should NOT have been called
        mock_llm.chat.assert_not_called()
