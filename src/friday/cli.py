"""CLI Interface for Friday."""

import asyncio
import logging
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .core.config import Config
from .core.agent_runner import AgentRunner
from .core.exceptions import ModelNotFoundError
from .utils.logging import setup_logging
from .voice.tts import TTSEngine
from .voice.stt import STTEngine

console = Console()
logger = logging.getLogger(__name__)

class FridayCLI:
    """The CLI interface and main event loop for Friday."""

    def __init__(self):
        self.config = Config()
        setup_logging(Path(self.config.get("logging.file")))
        
        self.runner = AgentRunner(self.config)
        self.tts = TTSEngine(self.config)
        self.stt = STTEngine(self.config)
        self.voice_mode = False

    async def run(self):
        """Run the main interactive loop."""
        console.print(Panel(
            "[bold green]FRIDAY SYSTEM ONLINE[/bold green]\n"
            "[dim]Type 'help' for commands, 'voice on/off' to toggle, or 'exit' to quit.[/dim]",
            title="Friday v0.2",
            subtitle="Local AI Assistant"
        ))

        while True:
            try:
                if self.voice_mode:
                    console.print("[dim italic]Listening...[/dim italic]")
                    user_input = await self.stt.listen()
                    if user_input:
                        console.print(f"[bold blue]You (Voice):[/bold blue] {user_input}")
                    else:
                        continue # Silent mic or timeout
                else:
                    user_input = Prompt.ask("[bold blue]>>>[/bold blue]")

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "bye"]:
                    await self.speak("Goodbye!")
                    break

                if user_input.lower() == "voice on":
                    self.voice_mode = True
                    await self.speak("Voice mode enabled.")
                    continue
                elif user_input.lower() == "voice off":
                    self.voice_mode = False
                    await self.speak("Voice mode disabled.")
                    continue

                # Process through the Runner
                response = await self.runner.handle_input(user_input)
                
                # Output to console and voice
                console.print(f"\n[bold green]Friday:[/bold green] {response}\n")
                await self.speak(response)

            except KeyboardInterrupt:
                break
            except ModelNotFoundError as e:
                console.print(f"[bold yellow]Voice Disabled:[/bold yellow] {e}")
                self.voice_mode = False
                console.print("[dim]Reverting to text mode. You can still use commands.[/dim]")
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                console.print(f"[bold red]System Error:[/bold red] {str(e)}")

    async def speak(self, text: str):
        """Handle both console and optional voice output."""
        import re
        # Clean text for TTS (remove all rich tags [tag]...[/tag] or [tag])
        clean_text = re.sub(r"\[.*?\]", "", text)
        
        try:
            await self.tts.speak(clean_text, block=False)
        except Exception:
            # Silent failure since text is already on screen
            pass


async def voice_download():
    """Download and extract voice models."""
    from .core.config import download_model
    import zipfile
    config = Config()
    
    models = {
        "TTS (Piper)": (
            "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx",
            Path(config.get("voice.tts.model_path"))
        ),
        "TTS Config": (
            "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json",
            Path(config.get("voice.tts.model_path")).with_suffix(".onnx.json")
        ),
        "STT (Vosk)": (
            "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
            Path(config.get("voice.stt.model_path")).with_suffix(".zip")
        )
    }

    try:
        console.print("[bold blue]Starting model downloads...[/bold blue]")
        for name, (url, dest) in models.items():
            try:
                # If model already exists, check if valid, otherwise skip
                if name.startswith("TTS"):
                    if dest.exists() and dest.stat().st_size > 0:
                        console.print(f"[dim]{name} already exists. Skipping.[/dim]")
                        continue

                console.print(f"Fetching [cyan]{name}[/cyan]...")
                download_model(url, dest)
                
                # Auto-extract STT
                if name == "STT (Vosk)":
                    console.print(f"Extracting [cyan]{name}[/cyan]...")
                    with zipfile.ZipFile(dest, 'r') as zip_ref:
                        zip_ref.extractall(dest.parent)
                    dest.unlink() # Cleanup zip
                    
            except Exception as e:
                console.print(f"[bold red]Failed to download {name}:[/bold red] {str(e)}")

        console.print("[bold green]Download and setup complete.[/bold green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Download interrupted by user.[/yellow]")

async def friday_doctor():
    """Perform a system health check."""
    config = Config()
    console.print(Panel("[bold blue]FRIDAY System Doctor[/bold blue]", expand=False))
    
    # 1. Check Configuration & Logs
    console.print(f"• Config: [green]OK[/green] ({config.config_path})")
    log_file = Path(config.get("logging.file"))
    if log_file.exists():
        mode = oct(log_file.stat().st_mode & 0o777)
        status = "[green]OK[/green]" if mode == "0o600" else f"[yellow]WARN (mode {mode})[/yellow]"
        console.print(f"• Logging: {status} ({log_file})")
    
    # 2. Check LLM (Ollama)
    from .llm.local import LocalEngine
    llm = LocalEngine(config.get("llm.primary_model"), config.get("llm.fallback_model"), config.get("llm.base_url"))
    if await llm.is_available_async():
        models = await llm.get_available_models()
        model_list = ", ".join(models) if models else "None"
        console.print(f"• LLM (Ollama): [green]ONLINE[/green] (Models: {model_list})")
    else:
        console.print("• LLM (Ollama): [bold red]OFFLINE[/bold red] (Is Ollama running?)")

    # 3. Check Voice Models
    tts_path = Path(config.get("voice.tts.model_path"))
    tts_status = "[green]EXISTS[/green]" if tts_path.exists() else "[red]MISSING[/red]"
    console.print(f"• TTS Model: {tts_status} ({tts_path.name})")
    
    stt_path = Path(config.get("voice.stt.model_path"))
    stt_status = "[green]EXISTS[/green]" if stt_path.exists() else "[red]MISSING[/red]"
    console.print(f"• STT Model: {stt_status} ({stt_path.name})")

    # 4. Check Audio Devices
    try:
        from .voice.stt import ignore_stderr
        with ignore_stderr():
            mics = STTEngine.list_microphones()
        if mics:
            console.print(f"• Audio Input: [green]OK[/green] ({len(mics)} devices found)")
        else:
            console.print("• Audio Input: [yellow]WARN[/yellow] (No microphones detected)")
    except Exception as e:
        console.print(f"• Audio Input: [red]ERROR[/red] ({e})")

async def main():
    """Main interactive loop."""
    cli = FridayCLI()
    await cli.run()

def app():
    """Entry point for the friday CLI as defined in pyproject.toml."""
    # Better arg matching
    args = [a.lower() for a in sys.argv[1:]]
    
    if "voice" in args and "download" in args:
        try:
            asyncio.run(voice_download())
            return
        except Exception as e:
            print(f"Download task failed: {e}")
            sys.exit(1)

    if "doctor" in args:
        try:
            asyncio.run(friday_doctor())
            return
        except Exception as e:
            print(f"Doctor failed: {e}")
            sys.exit(1)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    app()
