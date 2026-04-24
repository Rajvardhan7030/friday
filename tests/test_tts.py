"""Unit tests for the TTSEngine."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch

from friday.core.config import Config
from friday.voice.tts import TTSEngine


class SuccessfulProcess:
    """Simple subprocess stub for successful Piper execution."""

    returncode = 0

    async def communicate(self, input=None):
        return (b"stdout", b"stderr")


class HangingProcess:
    """Simple subprocess stub for timeout scenarios."""

    def __init__(self):
        self.kill = MagicMock()

    async def communicate(self, input=None):
        raise asyncio.TimeoutError()


async def _raise_timeout(awaitable, timeout):
    awaitable.close()
    raise asyncio.TimeoutError()


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = MagicMock(spec=Config)
    config.get.side_effect = lambda key, default=None: {
        "voice.tts.model": "en_GB-vits-low",
        "voice.tts.model_path": "/tmp/test_model.onnx",
        "voice.tts.piper_path": "piper"
    }.get(key, default)
    return config


@pytest.mark.asyncio
async def test_speak_model_not_found(mock_config):
    """Test speak method when model is missing."""
    with patch("friday.voice.tts.Path.exists", return_value=False):
        engine = TTSEngine(mock_config)
        # Should not raise exception but print missing message
        await engine.speak("Hello world")


@pytest.mark.asyncio
async def test_speak_success(mock_config):
    """Test successful TTS synthesis and playback."""
    mock_process = SuccessfulProcess()
    
    with patch("friday.voice.tts.Path.exists", return_value=True), \
         patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec, \
         patch("friday.voice.tts.TTSEngine._play_audio") as mock_play:
        
        engine = TTSEngine(mock_config)
        await engine.speak("Hello world")
        
        # Verify Piper was called
        mock_exec.assert_called_once()
        # Verify playback was triggered
        mock_play.assert_called_once()


@pytest.mark.asyncio
async def test_speak_timeout(mock_config):
    """Test Piper synthesis timeout."""
    mock_process = HangingProcess()
    
    with patch("friday.voice.tts.Path.exists", return_value=True), \
         patch("asyncio.create_subprocess_exec", return_value=mock_process), \
         patch("asyncio.wait_for", side_effect=_raise_timeout):
        
        engine = TTSEngine(mock_config)
        await engine.speak("Hello world")
        
        # Verify process was killed
        mock_process.kill.assert_called_once()
