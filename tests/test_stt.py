"""Unit tests for the STTEngine."""

import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock

from friday.core.config import Config
from friday.voice.stt import STTEngine

@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = MagicMock(spec=Config)
    config.get.side_effect = lambda key, default=None: {
        "voice.stt.model_path": "/tmp/test_stt_model",
        "voice.stt.samplerate": 16000,
        "voice.stt.device_index": 0,
        "voice.stt.energy_threshold": 300,
        "voice.stt.timeout": 1,
        "voice.stt.silence_limit": 0.5
    }.get(key, default)
    return config

@pytest.mark.asyncio
async def test_stt_listen_success(mock_config):
    """Test successful STT listening and transcription."""
    
    # 1. Mock PyAudio and Stream
    mock_pyaudio_inst = MagicMock()
    mock_stream = MagicMock()
    mock_pyaudio_inst.open.return_value = mock_stream
    
    # Simulate available data: first call returns 4000, second 4000, others 0
    mock_stream.get_read_available.side_effect = [4000, 4000] + [0] * 200
    mock_stream.read.return_value = b"\x00\x00" * 2000 # dummy audio
    
    # 2. Mock Vosk Model and Recognizer
    mock_model = MagicMock()
    mock_recognizer = MagicMock()
    
    # Simulate recognition
    mock_recognizer.AcceptWaveform.return_value = True
    mock_recognizer.Result.return_value = json.dumps({"text": "hello"})
    mock_recognizer.FinalResult.return_value = json.dumps({"text": "world"})
    
    with patch("friday.voice.stt.pyaudio.PyAudio", return_value=mock_pyaudio_inst), \
         patch("friday.voice.stt.vosk.Model", return_value=mock_model), \
         patch("friday.voice.stt.vosk.KaldiRecognizer", return_value=mock_recognizer), \
         patch("friday.voice.stt.Path.exists", return_value=True), \
         patch("friday.voice.stt.STTEngine._is_speech", return_value=True):
        
        engine = STTEngine(mock_config)
        
        # We need to mock asyncio.to_thread because we want to control its return value
        # or just let it run if the mocks above are thread-safe.
        # Since we are using MagicMocks, they are generally thread-safe for basic usage.
        
        result = await engine.listen()
        
        assert "hello" in result
        assert "world" in result
        
        # Verify to_thread was likely used (indirectly by checking mocks)
        assert mock_stream.read.called
        assert mock_recognizer.AcceptWaveform.called

@pytest.mark.asyncio
async def test_stt_timeout(mock_config):
    """Test STT timeout."""
    mock_pyaudio_inst = MagicMock()
    mock_stream = MagicMock()
    mock_pyaudio_inst.open.return_value = mock_stream
    
    # Never have data available
    mock_stream.get_read_available.return_value = 0
    
    with patch("friday.voice.stt.pyaudio.PyAudio", return_value=mock_pyaudio_inst), \
         patch("friday.voice.stt.vosk.Model", MagicMock()), \
         patch("friday.voice.stt.vosk.KaldiRecognizer", MagicMock()), \
         patch("friday.voice.stt.Path.exists", return_value=True):
        
        engine = STTEngine(mock_config)
        # Timeout is 1s in mock_config
        result = await engine.listen()
        
        assert result is None
