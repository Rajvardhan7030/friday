from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, Input, Log, Static
import asyncio
from friday.core.agent import UserMessage, FridayAgent

class AgentTUI(App):
    CSS = """
    #left-pane {
        width: 70%;
        height: 100%;
    }
    #right-pane {
        width: 30%;
        height: 100%;
        background: $boost;
    }
    #chat-log {
        height: 1fr;
    }
    #input-box {
        dock: bottom;
    }
    """

    def __init__(self, agent: FridayAgent):
        super().__init__()
        self.agent = agent

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="left-pane"):
                yield Log(id="chat-log")
                yield Input(placeholder="Type a message...", id="input-box")
            with Vertical(id="right-pane"):
                yield Static("FRIDAY Status\n---\nSystem Online", id="status-panel")
        yield Footer()

    async def on_input_submitted(self, event: Input.Submitted):
        if not event.value.strip():
            return
            
        input_box = self.query_one(Input)
        chat_log = self.query_one("#chat-log", Log)
        
        user_text = event.value
        input_box.value = ""
        chat_log.write_line(f"You: {user_text}")
        
        message = UserMessage(
            content=user_text,
            platform="terminal",
            user_id="local_user",
            timestamp=asyncio.get_event_loop().time()
        )
        
        try:
            response = await self.agent.process(message)
            chat_log.write_line(f"FRIDAY: {response}")
        except Exception as e:
            chat_log.write_line(f"[Error]: {e}")
