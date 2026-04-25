import asyncio
import os
import yaml
import logging
from friday.core.memory import ConversationMemory, SemanticMemory
from friday.core.skill_manager import SkillManager
from friday.core.agent import FridayAgent
from friday.providers.openai_provider import OpenAIProvider
from friday.providers.openrouter_provider import OpenRouterProvider
from friday.providers.ollama_provider import OllamaProvider
from friday.gateways.telegram_bot import TelegramGateway
from friday.gateways.discord_bot import DiscordGateway
from friday.gateways.slack_bot import SlackGateway
from friday.core.scheduler import TaskScheduler
from friday.interface.cli import cli

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config():
    config_path = os.path.expanduser("~/.friday/config.yaml")
    if not os.path.exists(config_path):
        local_path = "friday/config/settings.yaml"
        if os.path.exists(local_path):
            with open(local_path, "r") as f:
                return yaml.safe_load(f)
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        pass
    return {"llm": {"provider": "openrouter", "model": "google/gemini-2.5-flash-preview"}}

async def run_gateways(use_telegram=False, use_discord=False, use_slack=False):
    config = load_config()
    llm_conf = config.get("llm", {})
    provider_name = llm_conf.get("provider", "openrouter")
    
    if provider_name == "openai":
        provider = OpenAIProvider(api_key=llm_conf.get("api_key", ""), default_model=llm_conf.get("model"))
    elif provider_name == "ollama":
        provider = OllamaProvider(default_model=llm_conf.get("model"), base_url=llm_conf.get("base_url"))
    else:
        provider = OpenRouterProvider(api_key=llm_conf.get("api_key", ""), default_model=llm_conf.get("model"))
        
    conv_mem = ConversationMemory()
    await conv_mem.init_db()
    sem_mem = SemanticMemory()
    skill_mgr = SkillManager()
    
    agent = FridayAgent(provider, conv_mem, sem_mem, skill_mgr)
    
    gateways = []
    gateways_conf = config.get("gateways", {})
    
    if use_telegram or gateways_conf.get("telegram", {}).get("enabled"):
        token = gateways_conf.get("telegram", {}).get("token", os.environ.get("TELEGRAM_BOT_TOKEN", ""))
        if token:
            gw = TelegramGateway(token)
            gw.register_handler(agent.process)
            gateways.append(gw)
            
    if use_discord or gateways_conf.get("discord", {}).get("enabled"):
        token = gateways_conf.get("discord", {}).get("token", os.environ.get("DISCORD_BOT_TOKEN", ""))
        if token:
            gw = DiscordGateway(token)
            gw.register_handler(agent.process)
            gateways.append(gw)
            
    if use_slack or gateways_conf.get("slack", {}).get("enabled"):
        bot_token = gateways_conf.get("slack", {}).get("bot_token", os.environ.get("SLACK_BOT_TOKEN", ""))
        app_token = gateways_conf.get("slack", {}).get("app_token", os.environ.get("SLACK_APP_TOKEN", ""))
        if bot_token and app_token:
            gw = SlackGateway(bot_token, app_token)
            gw.register_handler(agent.process)
            gateways.append(gw)

    scheduler = TaskScheduler()
    scheduler.start()

    logger.info("Starting gateways...")
    await asyncio.gather(*(gw.start() for gw in gateways))
    
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down gateways...")
        await asyncio.gather(*(gw.stop() for gw in gateways))

if __name__ == "__main__":
    cli()
