"""CLI Interface for Friday."""

import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# Model Scout Imports
from .core.agent_runner import AgentRunner
from .core.config import Config
from .core.exceptions import ModelNotFoundError
from .utils.logging import ignore_stderr, setup_logging
from .voice.stt import STTEngine
from .voice.tts import TTSEngine

console = Console()
logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "nomic-embed-text:latest"
OPENAI_BASE_URL = "https://api.openai.com/v1"

class FridayCLI:
    """The CLI interface and main event loop for Friday."""

    def __init__(self, voice_output_enabled: bool = False):
        self.config = Config()
        setup_logging(Path(self.config.get("logging.file")))
        
        self.runner = AgentRunner(self.config)
        self.tts = TTSEngine(self.config)
        self.stt = STTEngine(self.config)
        self.voice_mode = False
        self.voice_output_enabled = voice_output_enabled

    @staticmethod
    def parse_control_command(user_input: str) -> str | None:
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
                
                # Use agent-provided TTS content if available, otherwise use the full response
                tts_text = self.runner.last_tts_content or response
                await self.speak(tts_text)

            except KeyboardInterrupt:
                break
            except ModelNotFoundError as e:
                console.print(f"[bold yellow]Voice Disabled:[/bold yellow] {e}")
                self.voice_mode = False
                console.print("[dim]Reverting to text mode. You can still use commands.[/dim]")
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                console.print(f"[bold red]System Error:[/bold red] {str(e)}")

    async def speak(self, text: str, block: bool = False):
        """Handle both console and optional voice output."""
        if not self.voice_output_enabled:
            return

        import re
        from rich.text import Text

        # 1. Handle Markdown links first: [text](url) -> text
        # This prevents Rich from misinterpreting [text] as a markup tag
        clean_text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

        # 2. Strip Rich markup
        try:
            clean_text = Text.from_markup(clean_text).plain
        except Exception:
            clean_text = re.sub(r"\[.*?\]", "", clean_text)

        # 3. Remove Markdown code blocks
        clean_text = re.sub(r"```.*?```", "", clean_text, flags=re.DOTALL)
        
        # 4. Remove remaining URLs
        clean_text = re.sub(r"https?://\S+", "", clean_text)
        
        # 5. Remove bold/italic/inline-code markers
        clean_text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", clean_text)
        clean_text = re.sub(r"(\*|_)(.*?)\1", r"\2", clean_text)
        clean_text = re.sub(r"`([^`]+)`", r"\1", clean_text)
        
        # 6. Normalize whitespace and remove common TTS "stutter" characters
        # Replace ... and --- with space to avoid stuttering
        clean_text = clean_text.replace("...", " ").replace("---", " ")
        
        # 7. Remove any leftover brackets (e.g., citations [1], [source: x])
        clean_text = re.sub(r"\[.*?\]", "", clean_text)
        
        # 8. Final whitespace normalization
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        if not clean_text:
            return
        
        try:
            await self.tts.speak(clean_text, block=block)
        except Exception:
            # Silent failure since text is already on screen
            pass


def extract_voice_output_flag(args: list[str]) -> tuple[bool, list[str]]:
    """Split voice-output flags from the remaining CLI args."""
    remaining_args: list[str] = []
    voice_output_enabled = False

    for arg in args:
        if arg.lower() in {"-v", "--voice-output"}:
            voice_output_enabled = True
        else:
            remaining_args.append(arg)

    return voice_output_enabled, remaining_args


async def voice_download() -> bool:
    """Download and extract voice models based on configuration."""
    import zipfile

    from .core.config import download_model
    config = Config()
    
    selected_tts_model = config.get("voice.tts.model")
    tts_urls = config.get("voice.urls.tts", {})
    stt_url = config.get("voice.urls.stt")
    tts_hashes = config.get("voice.hashes.tts", {})
    stt_hash = config.get("voice.hashes.stt")
    
    url = tts_urls.get(selected_tts_model, "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx")
    
    models = {
        f"TTS ({selected_tts_model})": (
            url,
            Path(config.get("voice.tts.model_path")),
            tts_hashes.get(selected_tts_model)
        ),
        "TTS Config": (
            url + ".json",
            Path(config.get("voice.tts.model_path")).with_suffix(".onnx.json"),
            None
        ),
        "STT (Vosk)": (
            stt_url,
            Path(config.get("voice.stt.model_path")).with_suffix(".zip"),
            stt_hash
        )
    }

    success = True
    try:
        console.print("[bold blue]Starting model downloads...[/bold blue]")
        for name, (url, dest, expected_hash) in models.items():
            try:
                # If model already exists, check if valid, otherwise skip
                if name.startswith("TTS"):
                    if dest.exists() and dest.stat().st_size > 0:
                        console.print(f"[dim]{name} already exists. Skipping.[/dim]")
                        continue

                console.print(f"Fetching [cyan]{name}[/cyan]...")
                download_model(url, dest, expected_hash=expected_hash)
                
                # Auto-extract STT
                if name == "STT (Vosk)":
                    console.print(f"Extracting [cyan]{name}[/cyan]...")
                    with zipfile.ZipFile(dest, 'r') as zip_ref:
                        zip_ref.extractall(dest.parent)
                    dest.unlink() # Cleanup zip
                    
            except Exception as e:
                console.print(f"[bold red]Failed to download {name}:[/bold red] {str(e)}")
                success = False

        if success:
            console.print("[bold green]Download and setup complete.[/bold green]")
        return success
    except KeyboardInterrupt:
        console.print("\n[yellow]Download interrupted by user.[/yellow]")
        return False

async def ollama_pull(model_name: str) -> bool:
    """Pull an Ollama model with progress feedback."""
    try:
        import ollama
        client = ollama.AsyncClient()
        
        console.print(f"Pulling model [cyan]{model_name}[/cyan] from Ollama...")
        try:
            with console.status(f"[bold green]Downloading {model_name}...[/bold green]") as status:
                async for part in await client.pull(model_name, stream=True):
                    if 'status' in part:
                        status.update(f"[[bold cyan]Ollama[/bold cyan]] {part['status']}...")
            
            console.print(f"[bold green]Successfully pulled {model_name}[/bold green]")
            return True
        except Exception as e:
            console.print(f"[bold red]Failed to pull {model_name}:[/bold red] {e}")
            console.print(f"[dim]Try running 'ollama pull {model_name}' manually.[/dim]")
            return False
    except ImportError:
        console.print("[bold red]Ollama library not installed.[/bold red]")
        return False

async def friday_config(args: list[str]):
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

async def friday_init():
    """Interactive initialization and hardware auto-tuning."""
    from .tui.onboarding import OnboardingApp
    
    app = OnboardingApp()
    results = await app.run_async()
    
    if not results:
        console.print("[yellow]Initialization cancelled.[/yellow]")
        return

    config = Config()
    
    # 1. User Identity
    config.set("user.name", results["user_name"], save=False)
    
    # 2. LLM Backend
    engine = results["engine"]
    if engine == "ollama":
        config.set("llm.engine", "ollama", save=False)
        config.set("llm.provider", "ollama", save=False)
        config.set("llm.primary_model", results["model"], save=False)
        config.set("llm.base_url", results["base_url"], save=False)
    else:
        # API Engine (openai, gemini, etc.)
        config.set("llm.engine", "openai", save=False) # Internal engine type for API
        config.set("llm.provider", results["provider"], save=False)
        config.set("llm.api_key", results["api_key"], save=False)
        config.set("llm.primary_model", results["model"], save=False)
        config.set("llm.api_base_url", results["base_url"], save=False)
    
    config.set("llm.embedding_model", results.get("embedding_model", DEFAULT_EMBED_MODEL), save=False)
    
    # 3. Voice Output
    config.set("voice.output_enabled", results["voice_enabled"], save=False)
    
    # 4. Persistence
    config.save()
    console.print("\n[bold green]Configuration saved successfully![/bold green]")

    # 5. Asset Setup with Retry/Rollback
    from rich.prompt import Confirm
    
    while True:
        setup_success = True
        
        if engine == "ollama":
            if not await ollama_pull(results["model"]):
                setup_success = False
        
        if results["voice_enabled"]:
            if not await voice_download():
                setup_success = False
                
        if setup_success:
            break
            
        console.print("\n[bold red]Critical asset setup failed.[/bold red]")
        if Confirm.ask("Would you like to retry the setup?"):
            continue
        
        if Confirm.ask("Abort initialization and rollback configuration?"):
            if config.config_path.exists():
                config.config_path.unlink()
                console.print("[yellow]Configuration rolled back (deleted).[/yellow]")
            return
        else:
            console.print("[yellow]Continuing with incomplete configuration. Some features may not work.[/yellow]")
            break

    console.print(Panel(
        f"[bold green]FRIDAY IS READY[/bold green]\n"
        f"Welcome, [cyan]{results['user_name']}[/cyan]! Type [cyan]friday[/cyan] to begin.",
        expand=False
    ))


async def friday_status():
    """Display a concise summary of the current configuration and system health."""
    config = Config()
    from rich.table import Table
    
    table = Table(title="FRIDAY System Status", show_header=False, box=None)
    table.add_row("[bold]Engine[/bold]")
    table.add_row(f"  Type:      [cyan]{config.get('llm.engine')}[/cyan]")
    table.add_row(f"  Provider:  [cyan]{config.get('llm.provider', 'N/A')}[/cyan]")
    
    table.add_row("\n[bold]Models[/bold]")
    table.add_row(f"  Primary:   [green]{config.get('llm.primary_model')}[/green]")
    table.add_row(f"  Fallback:  [green]{config.get('llm.fallback_model')}[/green]")
    table.add_row(f"  Embedding: [green]{config.get('llm.embedding_model')}[/green]")
    
    table.add_row("\n[bold]Voice[/bold]")
    table.add_row(f"  TTS Model: [cyan]{config.get('voice.tts.model')}[/cyan]")
    
    table.add_row("\n[bold]Memory[/bold]")
    status = "[green]Enabled[/green]" if config.get("memory.enabled") else "[red]Disabled[/red]"
    table.add_row(f"  Long-term: {status}")
    
    console.print(Panel(table, expand=False))

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
    
    # 2. Check LLM
    engine_type = config.get("llm.engine")
    if engine_type == "openai":
        from .llm.api import APIEngine
        llm = APIEngine(config.get("llm.primary_model"), config.get("llm.api_key"), config.get("llm.api_base_url"))
        if await llm.is_available_async():
            console.print(f"• LLM (API): [green]ONLINE[/green] (Model: {llm.model_name})")
        else:
            console.print("• LLM (API): [bold red]OFFLINE[/bold red] (Check API key or connectivity)")
    else:
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

    # 4. Check Browser Daemon
    try:
        import httpx
        daemon_url = config.get("skills.browser.daemon_url", "http://localhost:9000")
        try:
            resp = httpx.get(f"{daemon_url}/health", timeout=2.0)
            if resp.status_code == 200:
                console.print(f"• Browser Daemon: [green]ONLINE[/green] ({daemon_url})")
            else:
                console.print(f"• Browser Daemon: [yellow]ERROR[/yellow] (Status: {resp.status_code})")
        except Exception:
            console.print(f"• Browser Daemon: [bold red]OFFLINE[/bold red] (Is friday-browser-daemon running?)")
    except ImportError:
        pass

    # 5. Check Audio Devices
    try:
        from .utils.logging import ignore_stderr
        with ignore_stderr():
            mics = STTEngine.list_microphones()
        if mics:
            console.print(f"• Audio Input: [green]OK[/green] ({len(mics)} devices found)")
        else:
            console.print("• Audio Input: [yellow]WARN[/yellow] (No microphones detected)")
    except Exception as e:
        console.print(f"• Audio Input: [red]ERROR[/red] ({e})")

async def friday_model_scout(args: list[str]):
    """Run the model-scout tool."""
    import argparse

    from friday_model_scout.cli import run_scout

    parser = argparse.ArgumentParser(prog="friday model-scout", description="Friday Model Scout")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--filter", type=str)
    parser.add_argument("--sort", type=str, default="score")
    parser.add_argument("--ollama-only", action="store_true")

    # Filter out 'model-scout' from args if present (it shouldn't be here but just in case)
    clean_args = [a for a in args if a != "model-scout"]
    parsed_args = parser.parse_args(clean_args)
    
    await run_scout(
        json_output=parsed_args.json,
        filter_tag=parsed_args.filter,
        sort_by=parsed_args.sort,
        ollama_only=parsed_args.ollama_only
    )

async def main(voice_output_enabled: bool = False):
    """Main interactive loop."""
    cli = FridayCLI(voice_output_enabled=voice_output_enabled)
    await cli.run()


async def ask_mode(args: list[str], voice_output_enabled: bool = False):
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
    
    # Use agent-provided TTS content if available, otherwise use the full response
    tts_text = cli.runner.last_tts_content or response
    await cli.speak(tts_text, block=True)

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

    if normalized_args[:1] == ["init"]:
        try:
            asyncio.run(friday_init())
            return
        except Exception as e:
            print(f"Init failed: {e}")
            sys.exit(1)

    if normalized_args[:1] == ["status"]:
        try:
            asyncio.run(friday_status())
            return
        except Exception as e:
            print(f"Status failed: {e}")
            sys.exit(1)

    if normalized_args[:1] == ["doctor"]:
        try:
            asyncio.run(friday_doctor())
            return
        except Exception as e:
            print(f"Doctor failed: {e}")
            sys.exit(1)

    if normalized_args[:1] == ["model-scout"]:
        try:
            asyncio.run(friday_model_scout(raw_args[1:]))
            return
        except Exception as e:
            print(f"Model-scout failed: {e}")
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
