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

def get_runtime():
    """Initialize core runtime components."""
    settings = FridaySettings.load()
    setup_logging(settings.config_dir / "friday.log", level=logging.INFO)
    
    llm = LocalEngine(
        primary_model=settings.llm.primary_model,
        fallback_model=settings.llm.fallback_model,
        base_url=settings.llm.ollama_base_url
    )
    
    registry = SkillRegistry(user_skills_dir=settings.config_dir / "skills")
    registry.discover_built_in()
    
    vector_store = VectorStore(str(settings.memory.vector_db_path), llm)
    conv_memory = ConversationMemory(str(settings.memory.sqlite_db_path))
    
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
    mode: str = typer.Option("chat", help="Agent mode: chat, research, code")
):
    """Ask FRIDAY a question."""
    async def _ask():
        settings, llm, registry, vector_store, conv_memory, runner = get_runtime()
        await conv_memory.initialize()
        
        # Register specialized agents
        if mode == "research":
            from friday.agents.research import ResearchAgent
            runner.register_agent(ResearchAgent(llm, vector_store))
        elif mode == "code":
            from friday.agents.code_assistant import CodeAssistantAgent
            runner.register_agent(CodeAssistantAgent(llm, settings.workspace_dir))
        else:
            # Default chat agent could be a simpler one or the runner itself
            # For v0.1 we'll use a basic chat agent
            from friday.agents.base import BaseAgent, Context, AgentResult
            class ChatAgent(BaseAgent):
                @property
                def name(self): return "chat"
                @property
                def description(self): return "Simple chat agent."
                async def run(self, ctx: Context):
                    res = await self.llm.chat([{"role": "user", "content": ctx.user_query}])
                    return AgentResult(content=res.content)
            runner.register_agent(ChatAgent(llm))
            
        with console.status(f"[bold green]FRIDAY is thinking ({mode} mode)...[/bold green]"):
            result = await runner.run_agent(mode if mode != "chat" else "chat", query)
            
        console.print(f"\n[bold blue]FRIDAY:[/bold blue]\n{result.content}")
        
    asyncio.run(_ask())

@app.command()
def digest():
    """Run morning briefing."""
    async def _digest():
        settings, llm, registry, vector_store, conv_memory, runner = get_runtime()
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
    settings, llm, registry, vector_store, conv_memory, runner = get_runtime()
    
    table = Table(title="FRIDAY Diagnostics")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    
    # Check Ollama
    ollama_ok = llm.is_available()
    table.add_row("Ollama", "[green]Online[/green]" if ollama_ok else "[red]Offline[/red]")
    
    # Check Config
    config_ok = settings.config_dir.exists()
    table.add_row("Config Directory", "[green]OK[/green]" if config_ok else "[red]Missing[/red]")
    
    # Check Skills
    skills_count = len(registry.list_skills())
    table.add_row("Skills Loaded", f"[green]{skills_count}[/green]")
    
    console.print(table)

@app.command()
def memory(
    action: str = typer.Argument(..., help="Action: index, clear"),
    path: Optional[str] = typer.Option(None, help="Path to index")
):
    """Manage FRIDAY's long-term memory."""
    async def _memory():
        settings, llm, registry, vector_store, conv_memory, runner = get_runtime()
        indexer = DocumentIndexer(vector_store)
        
        if action == "index" and path:
            console.print(f"Indexing path: {path}...")
            count = await indexer.index_directory(Path(path))
            console.print(f"Indexed {count} chunks.")
        elif action == "clear":
            # For v0.1 reset vector store
            vector_store.client.reset()
            console.print("Memory cleared.")
            
    asyncio.run(_memory())

if __name__ == "__main__":
    app()
