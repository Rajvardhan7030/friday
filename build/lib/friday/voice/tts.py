"""Local TTS using Piper or Coqui."""

import os
import subprocess
import logging
import tempfile
from pathlib import Path
from typing import Optional
from pydub import AudioSegment
from pydub.playback import play

logger = logging.getLogger(__name__)

class TTSEngine:
    """Local text-to-speech engine."""

    def __init__(self, voice_model: str = "en_GB-vits-low"):
        self.voice_model = voice_model
        # Assume piper is in path or in ~/.friday/piper/
        self.piper_path = os.environ.get("PIPER_PATH", "piper")

    async def speak(self, text: str) -> None:
        """Synthesize text and play it."""
        import asyncio
        output_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_file = f.name

            # Run piper as an async subprocess
            process = await asyncio.create_subprocess_exec(
                self.piper_path, "--model", self.voice_model, "--output_file", output_file,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate(input=text.encode())
            
            if process.returncode == 0 and os.path.exists(output_file):
                # Play audio in a separate thread to avoid blocking event loop
                await asyncio.to_thread(self._play_audio, output_file)
            else:
                logger.error(f"TTS synthesis failed: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"TTS Engine error: {e}")
            print(f"\n{text}\n") # Fallback to printing if TTS fails
        finally:
            # Guarantee cleanup
            if output_file and os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except Exception as e:
                    logger.warning(f"Failed to cleanup TTS file {output_file}: {e}")

    def _play_audio(self, file_path: str) -> None:
        """Synchronous audio playback helper."""
        try:
            audio = AudioSegment.from_wav(file_path)
            play(audio)
        except Exception as e:
            logger.error(f"Playback error: {e}")
