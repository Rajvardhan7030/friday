import logging
import time
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from .base import BaseGateway
from friday.core.agent import UserMessage
import asyncio

logger = logging.getLogger(__name__)

class SlackGateway(BaseGateway):
    def __init__(self, bot_token: str, app_token: str):
        super().__init__()
        self.app = AsyncApp(token=bot_token)
        self.app_token = app_token
        self.handler = None
        self._setup_events()
        self._task = None

    def _setup_events(self):
        @self.app.message(".*")
        async def handle_message(message, say):
            if not self.message_handler:
                return

            user_id = message.get("user")
            text = message.get("text")
            
            # Ignore bot messages
            if message.get("bot_id"):
                return

            user_message = UserMessage(
                content=text,
                platform="slack",
                user_id=user_id,
                timestamp=time.time()
            )
            
            try:
                response = await self.message_handler(user_message)
                await say(response)
            except Exception as e:
                logger.error(f"Error handling Slack message: {e}")
                await say("An error occurred.")

    async def start(self):
        self.handler = AsyncSocketModeHandler(self.app, self.app_token)
        self._task = asyncio.create_task(self.handler.start_async())
        logger.info("Slack gateway started.")

    async def stop(self):
        if self.handler:
            await self.handler.close_async()
        if self._task:
            self._task.cancel()
        logger.info("Slack gateway stopped.")

    async def send_message(self, user_id: str, text: str):
        await self.app.client.chat_postMessage(channel=user_id, text=text)
