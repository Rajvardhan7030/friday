import click
import asyncio
import os
import yaml
from friday.core.memory import ConversationMemory, SemanticMemory
from friday.core.skill_manager import SkillManager
from friday.core.agent import FridayAgent, UserMessage
from friday.providers.openai_provider import OpenAIProvider
from friday.providers.openrouter_provider import OpenRouterProvider
from friday.providers.ollama_provider import OllamaProvider
from friday.core.scheduler import TaskScheduler

def load_config():
    config_path = os.path.expanduser("~/.friday/config.yaml")
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        default_config = {
            "llm": {"provider": "openrouter", "model": "google/gemini-2.5-flash-preview", "api_key": "", "base_url": None}
        }
        with open(config_path, "w") as f:
            yaml.dump(default_config, f)
        return default_config
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_agent(config):
    llm_conf = config.get("llm", {})
    provider_name = llm_conf.get("provider", "openrouter")
    
    if provider_name == "openai":
        provider = OpenAIProvider(api_key=llm_conf.get("api_key", ""), default_model=llm_conf.get("model"))
    elif provider_name == "ollama":
        provider = OllamaProvider(default_model=llm_conf.get("model"), base_url=llm_conf.get("base_url"))
    else:
        provider = OpenRouterProvider(api_key=llm_conf.get("api_key", ""), default_model=llm_conf.get("model"))
        
    conv_mem = ConversationMemory()
    sem_mem = SemanticMemory()
    skill_mgr = SkillManager()
    
    asyncio.run(conv_mem.init_db())
    
    return FridayAgent(provider, conv_mem, sem_mem, skill_mgr)

@click.group()
def cli():
    """FRIDAY CLI"""
    pass

@cli.command()
def chat():
    """Launch the Terminal UI"""
    config = load_config()
    agent = get_agent(config)
    from friday.interface.tui import AgentTUI
    app = AgentTUI(agent)
    app.run()

@cli.command()
@click.argument('question')
def ask(question):
    """Ask a one-off question"""
    config = load_config()
    agent = get_agent(config)
    
    async def run_ask():
        msg = UserMessage(content=question, platform="cli", user_id="local_user", timestamp=0.0)
        response = await agent.process(msg)
        click.echo(response)
        
    asyncio.run(run_ask())

@cli.command()
@click.option('--telegram', is_flag=True, help="Start Telegram gateway")
@click.option('--discord', is_flag=True, help="Start Discord gateway")
@click.option('--slack', is_flag=True, help="Start Slack gateway")
def gateway(telegram, discord, slack):
    """Start the specified gateways"""
    from friday.main import run_gateways
    asyncio.run(run_gateways(telegram, discord, slack))

@cli.command()
def skill_list():
    """List all available skills"""
    skill_mgr = SkillManager()
    click.echo(skill_mgr.get_skill_list())

@cli.group()
def schedule():
    """Manage scheduled tasks"""
    pass

@schedule.command()
@click.argument('name')
@click.option('--cron', required=True, help="Cron expression (e.g., '0 9 * * *')")
@click.option('--prompt', required=True, help="Prompt to send to agent")
def add(name, cron, prompt):
    """Add a scheduled job"""
    scheduler = TaskScheduler()
    parts = cron.split()
    if len(parts) == 5:
        cron_kwargs = {
            'minute': parts[0], 'hour': parts[1], 'day': parts[2], 'month': parts[3], 'day_of_week': parts[4]
        }
        scheduler.add_job(name, lambda: print(f"Scheduled prompt: {prompt}"), cron_kwargs=cron_kwargs)
        click.echo(f"Added scheduled job '{name}' with cron '{cron}'")
    else:
        click.echo("Invalid cron format. Need 5 parts.")

if __name__ == '__main__':
    cli()
