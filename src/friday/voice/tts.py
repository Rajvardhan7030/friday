"""Local TTS using Piper."""

import os
import asyncio
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Union

from ..core.config import Config
from ..core.exceptions import ModelNotFoundError

logger = logging.getLogger(__name__)


class TTSEngine:
    """Local text-to-speech engine using Piper."""

    def __init__(self, config: Config):
        """Initialize TTS engine with configuration.

        Args:
            config (Config): Central configuration object.
        """
        self.config = config
        self.voice_model = config.get("voice.tts.model")
        self.model_path = Path(config.get("voice.tts.model_path"))
        self.piper_path = config.get("voice.tts.piper_path")

    def _validate_model(self) -> None:
        """Ensure the voice model exists on disk."""
        if not self.model_path.exists():
            raise ModelNotFoundError(
                model_name=self.voice_model,
                model_path=str(self.model_path),
                download_hint="Run 'friday voice download' to fetch missing models."
            )

    async def speak(self, text: str, block: bool = True) -> None:
        """Synthesize text and play it.

        Args:
            text (str): The text to speak.
            block (bool, optional): Whether to wait for playback to finish. Defaults to True.
        """
        try:
            self._validate_model()
        except ModelNotFoundError as e:
            logger.error(e)
            print(f"\n[TTS MISSING] {text}\n")
            return

        output_file = None
        try:
            # Create a temporary file for the WAV output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                output_file = Path(f.name)

            # Piper expects text via stdin and can output to a file
            # We use asyncio.wait_for to prevent hanging
            process = await asyncio.create_subprocess_exec(
                self.piper_path,
                "--model", str(self.model_path),
                "--output_file", str(output_file),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                # Write text to stdin and close it
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=text.encode('utf-8')),
                    timeout=15.0
                )
                
                if process.returncode != 0:
                    error_msg = stderr.decode().strip()
                    logger.error(f"Piper failed (code {process.returncode}): {error_msg}")
                    print(f"\n{text}\n")
                    return

            except asyncio.TimeoutExpired:
                process.kill()
                logger.error("Piper synthesis timed out.")
                print(f"\n{text}\n")
                return

            if output_file.exists():
                if block:
                    await asyncio.to_thread(self._play_audio, output_file)
                else:
                    # Run in background
                    asyncio.create_task(asyncio.to_thread(self._play_audio, output_file, cleanup=True))
                    output_file = None # Don't cleanup in finally if playing in background

        except Exception as e:
            logger.error(f"TTS Engine error: {e}")
            print(f"\n{text}\n")
        finally:
            if output_file and output_file.exists():
                try:
                    output_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup TTS file {output_file}: {e}")

    def _play_audio(self, file_path: Path, cleanup: bool = False) -> None:
        """Synchronous audio playback helper with fallbacks.

        Args:
            file_path (Path): Path to the WAV file.
            cleanup (bool): Whether to delete the file after playback.
        """
        try:
            # Primary: pydub
            try:
                from pydub import AudioSegment
                from pydub.playback import play
                audio = AudioSegment.from_wav(str(file_path))
                play(audio)
                return
            except ImportError:
                logger.debug("pydub not available, trying alternative playback methods.")
            except Exception as e:
                logger.warning(f"pydub playback failed: {e}")

            # Fallback 1: playsound
            try:
                from playsound import playsound
                playsound(str(file_path))
                return
            except Exception:
                pass

            # Fallback 2: simpleaudio
            try:
                import simpleaudio as sa
                wave_obj = sa.WaveObject.from_wave_file(str(file_path))
                play_obj = wave_obj.play()
                play_obj.wait_done()
                return
            except Exception:
                pass

            # Fallback 3: system command (aplay, afplay)
            for cmd in ["aplay", "afplay", "pw-play"]:
                if shutil.which(cmd):
                    import subprocess
                    subprocess.run([cmd, str(file_path)], stderr=subprocess.DEVNULL)
                    return

            logger.error("All audio playback methods failed.")

        finally:
            if cleanup and file_path.exists():
                try:
                    file_path.unlink()
                except:
                    pass
