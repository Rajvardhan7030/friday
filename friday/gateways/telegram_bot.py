import logging
import time
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from .base import BaseGateway
from friday.core.agent import UserMessage

logger = logging.getLogger(__name__)

class TelegramGateway(BaseGateway):
    def __init__(self, token: str):
        super().__init__()
        self.token = token
        self.app = None

    async def start(self):
        self.app = ApplicationBuilder().token(self.token).build()
        self.app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self._handle_message))
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        logger.info("Telegram gateway started.")

    async def stop(self):
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            logger.info("Telegram gateway stopped.")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.message_handler:
            return

        user_id = str(update.message.from_user.id)
        text = update.message.text
        
        user_message = UserMessage(
            content=text,
            platform="telegram",
            user_id=user_id,
            timestamp=time.time()
        )
        
        try:
            response = await self.message_handler(user_message)
            await update.message.reply_text(response)
        except Exception as e:
            logger.error(f"Error handling Telegram message: {e}")
            await update.message.reply_text("An error occurred while processing your request.")

    async def send_message(self, user_id: str, text: str):
        if self.app and self.app.bot:
            await self.app.bot.send_message(chat_id=user_id, text=text)
