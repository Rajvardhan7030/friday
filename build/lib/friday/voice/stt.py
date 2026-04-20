"""Local Speech-to-Text using Vosk and PyAudio."""

import os
import json
import queue
import logging
import asyncio
from pathlib import Path
from typing import Optional

try:
    import pyaudio
    import vosk
except ImportError:
    pyaudio = None
    vosk = None

logger = logging.getLogger(__name__)

class STTEngine:
    """Local Speech-to-Text engine with silence detection."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model: Optional[vosk.Model] = None
        self._recognizer: Optional[vosk.KaldiRecognizer] = None
        self._q = queue.Queue()
        self.samplerate = 16000 # Standard for many STT models

    def _initialize_model(self) -> bool:
        """Load Vosk model if not already loaded."""
        if self._model:
            return True
        
        if not vosk:
            logger.error("Vosk not installed. Run 'pip install vosk'.")
            return False
            
        if not self.model_path or not Path(self.model_path).exists():
            logger.error(f"STT model not found at {self.model_path}. Please download it first.")
            return False
            
        try:
            self._model = vosk.Model(self.model_path)
            self._recognizer = vosk.KaldiRecognizer(self._model, self.samplerate)
            return True
        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}")
            return False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Put audio data into the queue."""
        self._q.put(bytes(in_data))
        return (None, pyaudio.paContinue)

    async def listen(self, timeout: int = 10, silence_limit: int = 2) -> Optional[str]:
        """Listen from microphone and return transcribed text."""
        if not self._initialize_model():
            return None
        
        if not pyaudio:
            logger.error("PyAudio not installed. Run 'pip install pyaudio'.")
            return None

        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.samplerate,
                input=True,
                frames_per_buffer=8000,
                stream_callback=self._audio_callback
            )
            
            logger.info("Listening...")
            stream.start_stream()
            
            full_text = ""
            last_speech_time = asyncio.get_event_loop().time()
            start_time = last_speech_time
            
            while stream.is_active():
                await asyncio.sleep(0.1)
                
                # Check for overall timeout
                if asyncio.get_event_loop().time() - start_time > timeout:
                    logger.info("STT timeout reached.")
                    break
                
                # Check for silence timeout
                if asyncio.get_event_loop().time() - last_speech_time > silence_limit and full_text:
                    logger.info("Silence detected. Stopping recording.")
                    break
                
                # Process audio queue
                while not self._q.empty():
                    data = self._q.get()
                    if self._recognizer.AcceptWaveform(data):
                        result = json.loads(self._recognizer.Result())
                        text = result.get("text", "").strip()
                        if text:
                            full_text += " " + text
                            last_speech_time = asyncio.get_event_loop().time()
                    else:
                        partial = json.loads(self._recognizer.PartialResult())
                        if partial.get("partial", "").strip():
                            last_speech_time = asyncio.get_event_loop().time()

            stream.stop_stream()
            stream.close()
            
            # Get final result
            final_res = json.loads(self._recognizer.FinalResult())
            full_text += " " + final_res.get("text", "").strip()
            
            return full_text.strip()
            
        except Exception as e:
            logger.error(f"STT Error: {e}")
            return None
        finally:
            p.terminate()
