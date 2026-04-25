import logging
import time
import discord
from .base import BaseGateway
from friday.core.agent import UserMessage
import asyncio

logger = logging.getLogger(__name__)

class DiscordGateway(BaseGateway):
    def __init__(self, token: str):
        super().__init__()
        self.token = token
        intents = discord.Intents.default()
        intents.message_content = True
        self.client = discord.Client(intents=intents)
        self._setup_events()
        self._task = None

    def _setup_events(self):
        @self.client.event
        async def on_ready():
            logger.info(f"Discord gateway started as {self.client.user}")

        @self.client.event
        async def on_message(message):
            if message.author == self.client.user:
                return

            if not self.message_handler:
                return

            user_id = str(message.author.id)
            user_message = UserMessage(
                content=message.content,
                platform="discord",
                user_id=user_id,
                timestamp=time.time()
            )
            
            try:
                response = await self.message_handler(user_message)
                await message.channel.send(response)
            except Exception as e:
                logger.error(f"Error handling Discord message: {e}")
                await message.channel.send("An error occurred.")

    async def start(self):
        self._task = asyncio.create_task(self.client.start(self.token))

    async def stop(self):
        if self.client:
            await self.client.close()
        if self._task:
            self._task.cancel()
        logger.info("Discord gateway stopped.")

    async def send_message(self, user_id: str, text: str):
        user = await self.client.fetch_user(int(user_id))
        if user:
            await user.send(text)
