"""Tests for CLI behavior."""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from friday import cli as cli_module
from friday.cli import FridayCLI, extract_voice_output_flag


@pytest.mark.asyncio
async def test_speak_skips_tts_when_voice_mode_disabled():
    cli = FridayCLI.__new__(FridayCLI)
    cli.voice_output_enabled = False
    cli.tts = MagicMock()
    cli.tts.speak = AsyncMock()

    await cli.speak("Hello world")

    cli.tts.speak.assert_not_called()


@pytest.mark.asyncio
async def test_speak_uses_tts_when_voice_mode_enabled():
    cli = FridayCLI.__new__(FridayCLI)
    cli.voice_output_enabled = True
    cli.tts = MagicMock()
    cli.tts.speak = AsyncMock()

    await cli.speak("[bold]Hello[/bold] world")

    cli.tts.speak.assert_awaited_once_with("Hello world", block=False)


def test_parse_control_command_requires_slash_prefix():
    assert FridayCLI.parse_control_command("voice on") is None
    assert FridayCLI.parse_control_command("/voice on") == "voice_on"
    assert FridayCLI.parse_control_command("/exit") == "exit"


def test_extract_voice_output_flag_supports_v_and_long_form():
    enabled, remaining = extract_voice_output_flag(["ask", "-v", "status"])
    assert enabled is True
    assert remaining == ["ask", "status"]

    enabled, remaining = extract_voice_output_flag(["--voice-output", "doctor"])
    assert enabled is True
    assert remaining == ["doctor"]


def test_app_routes_ask_command_with_voice_flag(monkeypatch):
    ask_mode = MagicMock()
    monkeypatch.setattr(cli_module, "ask_mode", ask_mode)
    monkeypatch.setattr(cli_module.asyncio, "run", MagicMock())
    monkeypatch.setattr(sys, "argv", ["friday", "ask", "-v", "Who", "are", "you?"])

    cli_module.app()

    cli_module.asyncio.run.assert_called_once()
    ask_mode.assert_called_once_with(
        ["Who", "are", "you?"],
        voice_output_enabled=True,
    )
