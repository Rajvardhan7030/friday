"""CLI Interface for Friday."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .core.config import Config
from .core.agent_runner import AgentRunner
from .core.exceptions import ModelNotFoundError
from .utils.logging import setup_logging, ignore_stderr
from .voice.tts import TTSEngine
from .voice.stt import STTEngine

console = Console()
logger = logging.getLogger(__name__)

class FridayCLI:
    """The CLI interface and main event loop for Friday."""

    def __init__(self, voice_output_enabled: bool = False):
        self.config = Config()
        setup_logging(Path(self.config.get("logging.file")))
        
        self.runner = AgentRunner(self.config)
        with ignore_stderr():
            self.tts = TTSEngine(self.config)
            self.stt = STTEngine(self.config)
        self.voice_mode = False
        self.voice_output_enabled = voice_output_enabled

    @staticmethod
    def parse_control_command(user_input: str) -> Optional[str]:
        """Return a normalized interactive control command, if present."""
        normalized = " ".join(user_input.strip().split()).lower()
        command_map = {
            "/exit": "exit",
            "/quit": "exit",
            "/bye": "exit",
            "/voice on": "voice_on",
            "/voice off": "voice_off",
        }
        return command_map.get(normalized)

    async def run(self):
        """Run the main interactive loop."""
        console.print(Panel(
            "[bold green]FRIDAY SYSTEM ONLINE[/bold green]\n"
            "[dim]Type 'help' for agent commands, '/voice on' to enable the mic, or '/exit' to quit.[/dim]",
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

                control_command = self.parse_control_command(user_input)

                if control_command == "exit":
                    await self.speak("Goodbye!")
                    break

                if control_command == "voice_on":
                    self.voice_mode = True
                    self.voice_output_enabled = True
                    await self.speak("Voice mode enabled.")
                    continue
                if control_command == "voice_off":
                    self.voice_mode = False
                    self.voice_output_enabled = False
                    console.print("[bold green]Friday:[/bold green] Voice mode disabled.\n")
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
        if not self.voice_output_enabled:
            return

        import re
        # Clean text for TTS (remove all rich tags [tag]...[/tag] or [tag])
        clean_text = re.sub(r"\[.*?\]", "", text)
        
        try:
            await self.tts.speak(clean_text, block=False)
        except Exception:
            # Silent failure since text is already on screen
            pass


def extract_voice_output_flag(args: List[str]) -> tuple[bool, List[str]]:
    """Split voice-output flags from the remaining CLI args."""
    remaining_args: List[str] = []
    voice_output_enabled = False

    for arg in args:
        if arg.lower() in {"-v", "--voice-output"}:
            voice_output_enabled = True
        else:
            remaining_args.append(arg)

    return voice_output_enabled, remaining_args


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

async def friday_config(args: List[str]):
    """Manage Friday configuration."""
    config = Config()
    
    if not args or args[0] == "list":
        from rich.table import Table
        table = Table(title="Friday Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        def add_rows(data, prefix=""):
            for k, v in data.items():
                key = f"{prefix}{k}"
                if isinstance(v, dict):
                    add_rows(v, f"{key}.")
                else:
                    table.add_row(key, str(v))
        
        add_rows(config.get_all())
        console.print(table)
        
    elif args[0] == "set" and len(args) >= 3:
        key, value = args[1], args[2]
        try:
            config.set(key, value)
            console.print(f"[bold green]Updated {key} to {value}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
    else:
        console.print("[yellow]Usage:[/yellow]")
        console.print("  friday config list")
        console.print("  friday config set <key> <value>")

async def friday_doctor():
# ... (rest of the file)
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

async def main(voice_output_enabled: bool = False):
    """Main interactive loop."""
    cli = FridayCLI(voice_output_enabled=voice_output_enabled)
    await cli.run()


async def ask_mode(args: List[str], voice_output_enabled: bool = False):
    """Run a single-prompt ask flow."""
    cli = FridayCLI(voice_output_enabled=voice_output_enabled)
    console.print(Panel(
        "[bold cyan]FRIDAY ASK MODE[/bold cyan]\n"
        "[dim]Ask one question and get a single response. Add -v to hear the answer.[/dim]",
        title="Friday Ask",
        subtitle="One-shot prompt"
    ))

    prompt = " ".join(args).strip() if args else Prompt.ask("[bold blue]Ask[/bold blue]").strip()
    if not prompt:
        console.print("[bold yellow]No question provided.[/bold yellow]")
        return

    response = await cli.runner.handle_input(prompt)
    console.print(f"\n[bold green]Friday:[/bold green] {response}\n")
    await cli.speak(response)

def app():
    """Entry point for the friday CLI as defined in pyproject.toml."""
    voice_output_enabled, raw_args = extract_voice_output_flag(sys.argv[1:])
    normalized_args = [arg.lower() for arg in raw_args]
    
    if normalized_args[:2] == ["voice", "download"]:
        try:
            asyncio.run(voice_download())
            return
        except Exception as e:
            print(f"Download task failed: {e}")
            sys.exit(1)

    if normalized_args[:1] == ["doctor"]:
        try:
            asyncio.run(friday_doctor())
            return
        except Exception as e:
            print(f"Doctor failed: {e}")
            sys.exit(1)
    
    if normalized_args[:1] == ["ask"]:
        try:
            asyncio.run(ask_mode(raw_args[1:], voice_output_enabled=voice_output_enabled))
            return
        except Exception as e:
            print(f"Ask command failed: {e}")
            sys.exit(1)

    if normalized_args[:1] == ["config"]:
        # Preserve original argument casing for values like file paths or model names.
        try:
            asyncio.run(friday_config(raw_args[1:]))
            return
        except Exception as e:
            print(f"Config command failed: {e}")
            sys.exit(1)
    
    try:
        asyncio.run(main(voice_output_enabled=voice_output_enabled))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    app()
