"""Typer-based CLI entrypoint for FRIDAY."""

import asyncio
import logging
import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import Optional

from friday.core.config import FridaySettings, DEFAULT_CONFIG_DIR
from friday.core.hardware import get_hardware_profile
from friday.core.agent_runner import AgentRunner
from friday.core.registry import SkillRegistry
from friday.llm.local import LocalEngine
from friday.memory.vector_store import VectorStore
from friday.memory.document_indexer import DocumentIndexer
from friday.memory.conversation import ConversationMemory
from friday.voice.tts import TTSEngine
from friday.utils.logging import setup_logging

app = typer.Typer(help="FRIDAY: Your local-first, privacy-centric AI assistant.")
console = Console()

async def get_runtime():
    """Initialize core runtime components asynchronously."""
    settings = FridaySettings.load()
    setup_logging(settings.config_dir / "friday.log", level=logging.INFO)
    
    llm = LocalEngine(
        primary_model=settings.llm.primary_model,
        fallback_model=settings.llm.fallback_model,
        base_url=settings.llm.ollama_base_url
    )
    
    registry = SkillRegistry(user_skills_dir=settings.config_dir / "skills")
    registry.discover_built_in()
    registry.discover_user_skills()
    
    vector_store = VectorStore(str(settings.memory.vector_db_path), llm)
    conv_memory = ConversationMemory(str(settings.memory.sqlite_db_path))
    
    await vector_store.initialize()
    await conv_memory.initialize()
    
    runner = AgentRunner(llm, registry)
    
    return settings, llm, registry, vector_store, conv_memory, runner

@app.command()
def init():
    """Initialize hardware, config, and models."""
    console.print("[bold blue]Initializing FRIDAY...[/bold blue]")
    
    # 1. Hardware detection
    profile = get_hardware_profile()
    console.print(f"Hardware detected: {profile.os}, {profile.cpu_cores} cores, {profile.ram_gb:.1f}GB RAM")
    
    recommended = profile.recommend_model()
    console.print(f"Recommended model based on your hardware: [bold green]{recommended}[/bold green]")
    
    # 2. Create config
    settings = FridaySettings()
    settings.llm.primary_model = recommended
    settings.save()
    console.print(f"Configuration saved to {DEFAULT_CONFIG_DIR}")
    
    # 3. Model Pull instructions
    console.print(f"\n[bold yellow]Next steps:[/bold yellow]")
    console.print(f"1. Install Ollama if you haven't already: https://ollama.ai/")
    console.print(f"2. Pull the recommended model: [cyan]ollama pull {recommended}[/cyan]")
    console.print(f"3. Run [cyan]friday doctor[/cyan] to verify installation.")

@app.command()
def ask(
    query: str,
    mode: str = typer.Option("chat", help="Agent mode: chat, research, code"),
    voice: bool = typer.Option(False, "--voice", "-v", help="Enable voice output")
):
    """Ask FRIDAY a question."""
    async def _ask():
        # Security: Basic Prompt Injection Detection
        injection_markers = ["ignore previous", "system prompt", "dan mode", "you are now"]
        if any(marker in query.lower() for marker in injection_markers):
            console.print("[bold red]Error: Potential prompt injection detected in query.[/bold red]")
            raise typer.Exit(code=1)

        settings, llm, registry, vector_store, conv_memory, runner = await get_runtime()
        
        # 1. Fetch history for context
        session_id = "default" # TODO: Support multiple sessions
        history = await conv_memory.get_history(session_id, limit=10)
        
        # 2. Register specialized agents
        if mode == "research":
            from friday.agents.research import ResearchAgent
            runner.register_agent(ResearchAgent(llm, vector_store))
        elif mode == "code":
            from friday.agents.code_assistant import CodeAssistantAgent
            runner.register_agent(CodeAssistantAgent(llm, settings.workspace_dir))
        else:
            from friday.agents.base import BaseAgent, Context, AgentResult
            class ChatAgent(BaseAgent):
                @property
                def name(self): return "chat"
                @property
                def description(self): return "Simple chat agent."
                async def run(self, ctx: Context):
                    # Construct prompt with System Message for stability
                    messages = [
                        {"role": "system", "content": f"You are {settings.persona_name}, a helpful, concise AI assistant. Always reply directly and avoid long, unrelated templates or irrelevant content unless specifically asked."}
                    ]
                    # Add history
                    messages.extend(ctx.chat_history)
                    # Add current query
                    messages.append({"role": "user", "content": ctx.user_query})
                    
                    res = await self.llm.chat(messages)
                    return AgentResult(content=res.content)
            runner.register_agent(ChatAgent(llm))
            
        with console.status(f"[bold green]FRIDAY is thinking ({mode} mode)...[/bold green]"):
            # 3. Add user message to memory
            await conv_memory.add_message(session_id, "user", query)
            
            # 4. Run agent with history
            result = await runner.run_agent(mode if mode != "chat" else "chat", query, history=history)
            
            # 5. Add assistant message to memory
            await conv_memory.add_message(session_id, "assistant", result.content)
            
        console.print(f"\n[bold blue]{settings.persona_name}:[/bold blue]\n{result.content}")

        # 6. Speak if requested
        if voice:
            tts = TTSEngine(settings.persona_voice)
            # Use downloaded model if available
            model_path = settings.config_dir / "voices" / f"{settings.persona_voice}.onnx"
            if model_path.exists():
                tts.voice_model = str(model_path)
            
            with console.status("[bold cyan]FRIDAY is speaking...[/bold cyan]"):
                await tts.speak(result.content)
        
    asyncio.run(_ask())

@app.command()
def digest():
    """Run morning briefing."""
    async def _digest():
        settings, llm, registry, vector_store, conv_memory, runner = await get_runtime()
        tts = TTSEngine(settings.persona_voice)
        
        from friday.agents.morning_digest import MorningDigestAgent
        agent = MorningDigestAgent(llm, tts)
        
        console.print("[bold blue]Starting Morning Digest...[/bold blue]")
        result = await agent.run(None) # Context not needed for digest
        console.print(f"\n[bold green]Briefing completed.[/bold green]\n{result.content}")

    asyncio.run(_digest())

@app.command()
def doctor():
    """Run diagnostics to check system health."""
    async def _doctor():
        settings, llm, registry, vector_store, conv_memory, runner = await get_runtime()
        
        table = Table(title="FRIDAY Diagnostics")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Details", style="yellow")
        
        # Check Ollama
        try:
            ollama_ok = await asyncio.to_thread(llm.is_available)
            status = "[green]Online[/green]" if ollama_ok else "[red]Offline[/red]"
            details = f"Base URL: {llm.base_url}"
            
            if ollama_ok:
                available_models = await llm.get_available_models()
                primary = settings.llm.primary_model
                fallback = settings.llm.fallback_model
                
                def is_available(model_name):
                    if model_name in available_models:
                        return True
                    if ":" not in model_name:
                        return f"{model_name}:latest" in available_models
                    return False

                missing_primary = not is_available(primary)
                missing_fallback = not is_available(fallback)
                
                if missing_primary and missing_fallback:
                    status = "[red]Critical[/red]"
                    details += f"\n[red]Both primary and fallback models missing: {primary}, {fallback}[/red]"
                    details += "\nRun 'ollama pull <model>' to fix or 'friday init' to re-detect hardware."
                elif missing_primary:
                    status = "[yellow]Degraded[/yellow]"
                    details += f"\n[red]Primary model missing: {primary}[/red]"
                    details += f"\n[green]Fallback model {fallback} is available.[/green]"
                    details += f"\nRun 'ollama pull {primary}' to fix."
                elif missing_fallback:
                    # If fallback is missing but primary is fine, it's just a warning
                    status = "[yellow]Warning[/yellow]"
                    details += f"\n[red]Fallback model missing: {fallback}[/red]"
                    details += "\n[green]Primary model is healthy.[/green]"
                    details += "\nRun 'friday init' to update defaults or pull the missing model."
                elif not available_models:
                    status = "[yellow]No Models[/yellow]"
                    details += "\n[red]No models found in Ollama.[/red]"
        except Exception as e:
            status = "[red]Error[/red]"
            details = str(e)
        table.add_row("Ollama", status, details)
        
        # Check Config
        config_ok = settings.config_dir.exists()
        table.add_row("Config Directory", "[green]OK[/green]" if config_ok else "[red]Missing[/red]", str(settings.config_dir))
        
        # Check Skills
        skills_count = len(registry.list_skills())
        table.add_row("Skills Loaded", f"[green]{skills_count}[/green]", ", ".join([s["name"] for s in registry.list_skills()[:5]]))
        
        console.print(table)
    
    asyncio.run(_doctor())

# Skill Management Subcommands
skill_app = typer.Typer(help="Manage FRIDAY's skills.")
app.add_typer(skill_app, name="skill")

@skill_app.command("list")
def list_skills():
    """List all available skills."""
    async def _list():
        _, _, registry, _, _, _ = await get_runtime()
        skills = registry.list_skills()
        
        table = Table(title="Available Skills")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        
        for skill in skills:
            table.add_row(skill["name"], skill["description"])
        
        console.print(table)
    asyncio.run(_list())

# Voice Management Subcommands
voice_app = typer.Typer(help="Manage FRIDAY's voice.")
app.add_typer(voice_app, name="voice")

@voice_app.command("download")
def download_voice(
    model_name: Optional[str] = typer.Option(None, help="Voice model name (e.g., en_GB-southern_english_female-low)")
):
    """Download a Piper voice model."""
    async def _download():
        settings = FridaySettings.load()
        model = model_name or settings.persona_voice
        voice_dir = settings.config_dir / "voices"
        voice_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[bold blue]Downloading voice model: {model}...[/bold blue]")
        
        import httpx
        from pathlib import Path
        
        # Piper repo structure: {lang_family}/{lang_code}/{voice_name}/{quality}/{model}.onnx
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
        
        # Robust parsing for model name like 'en_GB-southern_english_female-low'
        try:
            parts = model.split("-")
            lang_code = parts[0] # en_GB
            lang_family = lang_code.split("_")[0] # en
            quality = parts[-1] # low
            voice_name = "-".join(parts[1:-1]) # southern_english_female
            
            files = [f"{model}.onnx", f"{model}.onnx.json"]
            
            async with httpx.AsyncClient(follow_redirects=True) as client:
                for file in files:
                    url = f"{base_url}/{lang_family}/{lang_code}/{voice_name}/{quality}/{file}"
                    target = voice_dir / file
                    
                    if target.exists():
                        console.print(f"[yellow]File {file} already exists, skipping.[/yellow]")
                        continue
                    
                    console.print(f"Fetching {file}...")
                    response = await client.get(url)
                    if response.status_code == 200:
                        target.write_bytes(response.content)
                        console.print(f"[green]Successfully downloaded {file}[/green]")
                    else:
                        console.print(f"[red]Failed to download {file}: HTTP {response.status_code}[/red]")
                        console.print(f"URL tried: {url}")
        except Exception as e:
            console.print(f"[red]Error parsing model name: {e}[/red]")
            console.print("Ensure model name follows format: lang_CODE-voice_name-quality")

    asyncio.run(_download())

@voice_app.command("test")
def test_voice(text: str = typer.Argument("Hello, I am Friday. Your personal assistant.", help="The text to speak")):
    """Test the current voice configuration."""
    async def _test():
        settings, _, _, _, _, _ = await get_runtime()
        tts = TTSEngine(settings.persona_voice)
        # Check if model exists in voices dir
        model_path = settings.config_dir / "voices" / f"{settings.persona_voice}.onnx"
        if model_path.exists():
            tts.voice_model = str(model_path)
            
        console.print(f"[bold green]Speaking with voice: {settings.persona_voice}...[/bold green]")
        await tts.speak(text)
    
    asyncio.run(_test())

@app.command()
def memory(
    action: str = typer.Argument(..., help="Action: index, clear"),
    path: Optional[str] = typer.Option(None, help="Path to index")
):
    """Manage FRIDAY's long-term memory."""
    async def _memory():
        settings, llm, registry, vector_store, conv_memory, runner = await get_runtime()
        indexer = DocumentIndexer(vector_store)
        
        if action == "index" and path:
            console.print(f"Indexing path: {path}...")
            count = await indexer.index_directory(Path(path))
            console.print(f"Indexed {count} chunks.")
        elif action == "clear":
            # For v0.1 reset vector store
            if hasattr(vector_store.client, "reset"):
                vector_store.client.reset()
                console.print("Memory cleared.")
            else:
                console.print("[yellow]Warning: Vector store does not support reset.[/yellow]")
            
    asyncio.run(_memory())

if __name__ == "__main__":
    app()
