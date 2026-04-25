from abc import ABC, abstractmethod
from typing import Callable, Awaitable
from friday.core.agent import UserMessage

class BaseGateway(ABC):
    def __init__(self):
        self.message_handler = None

    def register_handler(self, handler: Callable[[UserMessage], Awaitable[str]]):
        self.message_handler = handler

    @abstractmethod
    async def start(self):
        pass

    @abstractmethod
    async def stop(self):
        pass

    @abstractmethod
    async def send_message(self, user_id: str, text: str):
        pass
