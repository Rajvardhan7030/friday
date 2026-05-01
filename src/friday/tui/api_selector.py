from __future__ import annotations
from typing import Optional
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Select, Input, Button, Label
from textual.containers import Container, Horizontal

class APISelectorApp(App[Optional[dict[str, str]]]):
    """TUI for selecting and configuring an API provider for FRIDAY."""
    
    TITLE = "FRIDAY API Selector"
    SUB_TITLE = "Configure your LLM Backend"

    CSS = """
    Screen {
        align: center middle;
    }

    Container {
        padding: 1 2;
        width: 70;
        height: auto;
        border: thick $primary;
        background: $surface;
    }

    Label {
        margin-top: 1;
        text-style: bold;
    }

    Input {
        margin-bottom: 1;
    }

    #buttons {
        margin-top: 2;
        align: right middle;
        height: auto;
    }

    Button {
        margin-left: 1;
    }
    """

    PROVIDERS = [
        ("Local Ollama", "ollama"),
        ("OpenAI", "openai"),
        ("Gemini", "gemini"),
        ("Mistral", "mistral"),
        ("Groq", "groq"),
        ("OpenRouter", "openrouter"),
        ("Other (OpenAI-compatible)", "other"),
    ]

    # Provider defaults: (Model Name, Base URL)
    DEFAULTS: dict[str, tuple[str, str]] = {
        "ollama": ("llama3", "http://localhost:11434/v1"),
        "openai": ("gpt-4o", "https://api.openai.com/v1"),
        "gemini": ("gemini-1.5-flash", "https://generativelanguage.googleapis.com/v1beta/openai"),
        "mistral": ("mistral-large-latest", "https://api.mistral.ai/v1"),
        "groq": ("llama3-70b-8192", "https://api.groq.com/openai/v1"),
        "openrouter": ("anthropic/claude-3.5-sonnet", "https://openrouter.ai/api/v1"),
        "other": ("gpt-4o", "https://api.openai.com/v1"),
    }

    def compose(self) -> ComposeResult:
        yield Header()
        with Container():
            yield Label("Provider Selection")
            yield Select(self.PROVIDERS, id="provider", value="ollama")
            
            yield Label("API Key", id="api_key_label")
            yield Input(placeholder="Enter your API Key", id="api_key", password=True)
            
            yield Label("Model Name")
            yield Input(placeholder="e.g. gpt-4o", id="model_name")
            
            yield Label("Base URL")
            yield Input(placeholder="https://...", id="base_url")
            
            with Horizontal(id="buttons"):
                yield Button("Cancel", id="cancel", variant="error")
                yield Button("Save", id="save", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the form with default values."""
        self.update_fields("ollama")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle provider selection changes."""
        if event.value:
            self.update_fields(str(event.value))

    def update_fields(self, provider: str) -> None:
        """Update input fields based on the selected provider."""
        model_name, base_url = self.DEFAULTS.get(provider, ("", ""))
        
        model_input = self.query_one("#model_name", Input)
        url_input = self.query_one("#base_url", Input)
        api_key_input = self.query_one("#api_key", Input)
        api_key_label = self.query_one("#api_key_label", Label)
        
        model_input.value = model_name
        url_input.value = base_url
        
        is_ollama = provider == "ollama"
        api_key_input.display = not is_ollama
        api_key_label.display = not is_ollama

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "save":
            provider = self.query_one("#provider", Select).value
            results = {
                "engine": "ollama" if provider == "ollama" else "openai",
                "api_key": self.query_one("#api_key", Input).value,
                "model_name": self.query_one("#model_name", Input).value,
                "base_url": self.query_one("#base_url", Input).value,
                "provider": str(provider)
            }
            self.exit(results)
        elif event.button.id == "cancel":
            self.exit(None)

if __name__ == "__main__":
    app = APISelectorApp()
    app.run()
