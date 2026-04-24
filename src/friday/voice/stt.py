"""Local Speech-to-Text using Vosk and PyAudio."""

from __future__ import annotations

import sys
import json
import logging
import asyncio
import os
import contextlib
from pathlib import Path
from typing import Any, Optional, List, Dict

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pyaudio
    import vosk
except ImportError:
    pyaudio = None
    vosk = None

from ..core.config import Config
from ..core.exceptions import ModelNotFoundError, AudioDeviceError

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def ignore_stderr():
    """Context manager to silence stderr (for noisy C libraries like PortAudio)."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        sys.stderr.flush()
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
    except OSError:
        # Fallback if dup/dup2 not available
        yield


class STTEngine:
    """Local Speech-to-Text engine with silence detection and energy VAD."""

    def __init__(self, config: Config):
        """Initialize STT engine with configuration.

        Args:
            config (Config): Central configuration object.
        """
        self.config = config
        self.model_path = Path(config.get("voice.stt.model_path"))
        self.samplerate = config.get("voice.stt.samplerate", 16000)
        self.device_index = config.get("voice.stt.device_index")
        self.energy_threshold = config.get("voice.stt.energy_threshold", 300)
        
        self._model: Optional[Any] = None
        self._recognizer: Optional[Any] = None
        self._audio_event = asyncio.Event()

    def _initialize_model(self) -> None:
        """Load Vosk model if not already loaded."""
        if self._model:
            return

        if not vosk:
            raise ImportError("Vosk not installed. Run 'pip install vosk'.")

        if not self.model_path.exists():
            raise ModelNotFoundError(
                model_name="Vosk STT",
                model_path=str(self.model_path),
                download_hint="Run 'friday voice download' to fetch the STT model."
            )

        try:
            self._model = vosk.Model(str(self.model_path))
            self._recognizer = vosk.KaldiRecognizer(self._model, self.samplerate)
        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}")
            raise

    @classmethod
    def list_microphones(cls) -> List[Dict[str, any]]:
        """List available microphone devices."""
        if not pyaudio:
            return []
        
        p = pyaudio.PyAudio()
        info = []
        try:
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                if dev.get('maxInputChannels') > 0:
                    info.append({
                        'index': i,
                        'name': dev.get('name'),
                        'rate': int(dev.get('defaultSampleRate'))
                    })
        finally:
            p.terminate()
        return info

    def _is_speech(self, audio_data: bytes) -> bool:
        """Simple energy-based Voice Activity Detection (VAD)."""
        if not audio_data:
            return False
        if np is not None:
            # Convert bytes to int16 array when numpy is available.
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            energy = np.sqrt(np.mean(audio_np**2))
        else:
            import audioop
            energy = audioop.rms(audio_data, 2)
        return energy > self.energy_threshold

    async def listen(
        self, 
        timeout: Optional[int] = None, 
        silence_limit: Optional[int] = None
    ) -> Optional[str]:
        """Listen from microphone and return transcribed text.

        Args:
            timeout (int, optional): Overall listening timeout in seconds.
            silence_limit (int, optional): Seconds of silence before stopping.

        Returns:
            Optional[str]: Transcribed text or None.
        """
        self._initialize_model()
        
        timeout = timeout or self.config.get("voice.stt.timeout", 10)
        silence_limit = silence_limit or self.config.get("voice.stt.silence_limit", 2)

        if not pyaudio:
            logger.error("PyAudio not installed. Run 'pip install pyaudio'.")
            return None

        with ignore_stderr():
            p = pyaudio.PyAudio()
        
        stream = None
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.samplerate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=8000
            )

            logger.info("Listening...")
            full_text = ""
            last_speech_time = asyncio.get_event_loop().time()
            start_time = last_speech_time
            
            # Use non-blocking read in a loop
            while True:
                # 1. Read audio data (non-blocking if possible)
                if stream.get_read_available() > 0:
                    data = stream.read(4000, exception_on_overflow=False)
                    
                    if self._is_speech(data):
                        last_speech_time = asyncio.get_event_loop().time()
                        
                        if self._recognizer.AcceptWaveform(data):
                            result = json.loads(self._recognizer.Result())
                            text = result.get("text", "").strip()
                            if text:
                                full_text += " " + text
                        else:
                            # Partial results can be ignored unless we want real-time display
                            pass
                
                # 2. Check timeouts
                now = asyncio.get_event_loop().time()
                
                if now - start_time > timeout:
                    logger.info("STT overall timeout reached.")
                    break
                    
                if full_text and (now - last_speech_time > silence_limit):
                    logger.info("Silence detected. Stopping.")
                    break
                
                await asyncio.sleep(0.01) # Yield to event loop

            # Get final result
            final_res = json.loads(self._recognizer.FinalResult())
            full_text += " " + final_res.get("text", "").strip()
            
            transcription = full_text.strip()
            if transcription:
                logger.info(f"Transcribed: {transcription}")
            return transcription or None

        except Exception as e:
            logger.error(f"STT Error: {e}")
            return None
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()
