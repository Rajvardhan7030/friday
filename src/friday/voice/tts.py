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
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_file = f.name

            # Run piper as a subprocess
            # Command: echo text | piper --model model.onnx --output_file out.wav
            # v0.1: simple subprocess call. Requires piper binary and model to be setup.
            process = subprocess.Popen(
                [self.piper_path, "--model", self.voice_model, "--output_file", output_file],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode == 0 and os.path.exists(output_file):
                # Play audio
                audio = AudioSegment.from_wav(output_file)
                play(audio)
            else:
                logger.error(f"TTS synthesis failed: {stderr}")
                
            # Cleanup
            if os.path.exists(output_file):
                os.remove(output_file)
                
        except Exception as e:
            logger.error(f"TTS Engine error: {e}")
            print(f"\n{text}\n") # Fallback to printing if TTS fails
