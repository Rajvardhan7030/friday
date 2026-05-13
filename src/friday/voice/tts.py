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
        self._playback_lock = asyncio.Lock()

    def _validate_model(self) -> None:
        """Ensure the voice model and its JSON config exist and are valid."""
        import json
        config_path = self.model_path.with_suffix(".onnx.json")
        
        missing = []
        if not self.model_path.exists():
            missing.append(f"Model ({self.model_path.name})")
        
        if not config_path.exists():
            logger.warning("Piper config file %s is missing; continuing with model validation only.", config_path.name)
        else:
            try:
                with open(config_path, "r") as f:
                    json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Piper config file %s is unreadable or invalid JSON.", config_path.name)
            
        if missing:
            raise ModelNotFoundError(
                model_name=self.voice_model,
                model_path=", ".join(missing),
                download_hint="Run 'friday voice download' to fetch or repair models."
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

            # Calculate dynamic timeout: min 30s, plus 1s per 20 characters
            # This handles long responses on slower CPUs.
            synthesis_timeout = max(30.0, len(text) / 20.0)

            try:
                # Write text to stdin and close it
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=text.encode('utf-8')),
                    timeout=synthesis_timeout
                )
                
                if process.returncode != 0:
                    error_msg = stderr.decode().strip()
                    logger.error(f"Piper failed (code {process.returncode}): {error_msg}")
                    return

            except asyncio.TimeoutError:
                try:
                    process.kill()
                    await process.wait()
                except Exception:
                    pass
                logger.error(f"Piper synthesis timed out after {synthesis_timeout:.1f}s (text length: {len(text)})")
                # Fallback: print text if voice fails
                print(f"\n{text}\n")
                return

            if output_file.exists():
                if block:
                    async with self._playback_lock:
                        await asyncio.to_thread(self._play_audio, output_file)
                else:
                    # Run in background via a helper to ensure lock acquisition and cleanup
                    async def _play_and_cleanup(path: Path):
                        try:
                            async with self._playback_lock:
                                await asyncio.to_thread(self._play_audio, path)
                        finally:
                            if path.exists():
                                try:
                                    path.unlink()
                                except Exception as e:
                                    logger.warning(f"Failed to cleanup background TTS file {path}: {e}")

                    asyncio.create_task(_play_and_cleanup(output_file))
                    output_file = None # Ownership transferred to task

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

    async def aclose(self) -> None:
        """Close any open resources."""
        pass
