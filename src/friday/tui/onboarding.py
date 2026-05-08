from __future__ import annotations

import asyncio
import logging
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Select,
    Static,
    Switch,
)

from friday.core.config import Config
from friday.core.hardware import HardwareProfile, get_hardware_profile, get_recommended_model
from friday.tui.narratives import FRIDAY_ASCII, ONBOARDING
from friday_model_scout.compatibility_engine import get_compatible_models
from friday_model_scout.model_database import get_models

logger = logging.getLogger(__name__)

class WelcomeScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Container(
            Static(FRIDAY_ASCII, id="ascii-art"),
            Static(ONBOARDING["welcome"]["title"], id="welcome-title"),
            Static(ONBOARDING["welcome"]["text"], id="welcome-text"),
            Horizontal(
                Button(ONBOARDING["welcome"]["button"], id="initiate", variant="primary"),
                id="button-container"
            ),
            id="welcome-container"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "initiate":
            self.app.push_screen(BackendChoiceScreen())

class BackendChoiceScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Container(
            Label(ONBOARDING["backend"]["title"], id="backend-title"),
            Static(ONBOARDING["backend"]["text"], id="backend-desc"),
            Horizontal(
                Button(ONBOARDING["backend"]["local_btn"], id="btn-local", variant="primary"),
                Button(ONBOARDING["backend"]["cloud_btn"], id="btn-cloud"),
                id="button-container"
            ),
            id="backend-container"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-local":
            self.app.engine_type = "ollama"
            self.app.push_screen(HardwareScreen())
        elif event.button.id == "btn-cloud":
            self.app.engine_type = "api"
            self.app.push_screen(APIScreen())

class APIScreen(Screen):
    PROVIDERS = [
        ("OpenAI", "openai"),
        ("Gemini", "gemini"),
        ("Mistral", "mistral"),
        ("Groq", "groq"),
        ("OpenRouter", "openrouter"),
        ("Other (OpenAI-compatible)", "other"),
    ]

    def compose(self) -> ComposeResult:
        yield Container(
            Label(ONBOARDING["api"]["title"], id="api-title"),
            Label(ONBOARDING["api"]["provider_label"]),
            Select(self.PROVIDERS, id="provider", value="openai"),
            
            Label(ONBOARDING["api"]["key_label"], id="api_key_label"),
            Input(placeholder=ONBOARDING["api"]["key_placeholder"], id="api_key", password=True),
            
            Label(ONBOARDING["api"]["model_label"]),
            Input(placeholder=ONBOARDING["api"]["model_placeholder"], id="model_name"),
            
            Label(ONBOARDING["api"]["url_label"]),
            Input(placeholder=ONBOARDING["api"]["url_placeholder"], id="base_url"),
            
            Horizontal(
                Button(ONBOARDING["api"]["back_btn"], id="back"),
                Button(ONBOARDING["api"]["continue_btn"], id="continue", variant="primary"),
                id="button-container"
            ),
            id="api-container"
        )

    def on_mount(self) -> None:
        self.update_fields("openai")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.value:
            self.update_fields(str(event.value))

    def update_fields(self, provider: str) -> None:
        defaults = Config.PROVIDER_DEFAULTS.get(provider, {})
        self.query_one("#model_name", Input).value = defaults.get("model", "")
        self.query_one("#base_url", Input).value = defaults.get("url", "")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "continue":
            provider = str(self.query_one("#provider", Select).value)
            defaults = Config.PROVIDER_DEFAULTS.get(provider, {})
            
            self.app.api_results = {
                "engine": defaults.get("engine", "openai"),
                "api_key": self.query_one("#api_key", Input).value,
                "model": self.query_one("#model_name", Input).value,
                "base_url": self.query_one("#base_url", Input).value,
                "embedding_model": defaults.get("embedding", "text-embedding-3-small"),
                "provider": provider
            }
            self.app.selected_model = self.app.api_results["model"]
            self.app.push_screen(VoiceInterviewScreen())
        elif event.button.id == "back":
            self.app.pop_screen()

class HardwareScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Container(
            Label(ONBOARDING["hardware"]["scanning"], id="scan-label"),
            Vertical(
                Static("Detecting OS...", id="hw-os"),
                Static("Detecting CPU...", id="hw-cpu"),
                Static("Detecting RAM...", id="hw-ram"),
                Static("Detecting GPU...", id="hw-gpu"),
                id="hardware-panel"
            ),
            Horizontal(
                Button(ONBOARDING["hardware"]["continue_btn"], id="continue", variant="primary", disabled=True),
                id="button-container"
            ),
            id="hardware-container"
        )

    async def on_mount(self) -> None:
        try:
            profile = await get_hardware_profile()
        except Exception as e:
            import platform

            from friday.core.hardware import HardwareProfile
            logger.error(f"Hardware detection failed: {e}")
            # Safe fallback profile
            profile = HardwareProfile(
                os=platform.system(),
                cpu_arch=platform.machine(),
                cpu_cores=0,
                cpu_threads=0,
                ram_gb=4.0,
                gpu_vram_gb=None,
                gpu_name=None
            )
            self.query_one("#scan-label", Label).update(ONBOARDING["hardware"]["failure"])
        else:
            self.query_one("#scan-label", Label).update(ONBOARDING["hardware"]["success"])

        self.app.hardware_profile = profile
        
        self.query_one("#hw-os", Static).update(f"OS: [cyan]{profile.os}[/cyan]")
        self.query_one("#hw-cpu", Static).update(f"CPU: [cyan]{profile.cpu_arch}[/cyan] ({profile.cpu_cores} cores)")
        self.query_one("#hw-ram", Static).update(f"RAM: [cyan]{profile.ram_gb:.1f} GB[/cyan]")
        
        gpu_info = f"[cyan]{profile.gpu_name}[/cyan] ([green]{profile.gpu_vram_gb:.1f} GB VRAM[/green])" if profile.gpu_name else "[yellow]None detected (Using CPU)[/yellow]"
        self.query_one("#hw-gpu", Static).update(f"GPU: {gpu_info}")
        
        self.query_one("#continue", Button).disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "continue":
            self.app.push_screen(ModelSelectionScreen())

class ModelSelectionScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Container(
            Label(ONBOARDING["model"]["title"], id="model-title"),
            Static(ONBOARDING["model"]["desc"], id="model-desc"),
            RadioSet(
                RadioButton(ONBOARDING["model"]["opt_auto"].format(model="..."), id="opt-auto", value=True),
                RadioButton(ONBOARDING["model"]["opt_scout"].format(model="..."), id="opt-scout"),
                RadioButton(ONBOARDING["model"]["opt_custom"], id="opt-custom"),
                id="model-options"
            ),
            Vertical(
                Label(ONBOARDING["model"]["custom_label"]),
                Input(placeholder=ONBOARDING["model"]["custom_placeholder"], id="custom-model-input"),
                id="custom-container",
                classes="hidden"
            ),
            Horizontal(
                Button(ONBOARDING["model"]["confirm_btn"], id="confirm", variant="primary", disabled=True),
                id="button-container"
            ),
            id="model-container"
        )

    async def on_mount(self) -> None:
        profile: HardwareProfile = self.app.hardware_profile
        
        # Use asyncio.to_thread for potentially heavy Model Scout logic
        suggested = await asyncio.to_thread(get_recommended_model, profile)
        models = await asyncio.to_thread(get_models)
        scout_results = await asyncio.to_thread(get_compatible_models, models, profile.to_detailed())
        
        scout_results.sort(key=lambda x: x["compat"]["score"], reverse=True)
        ollama_results = [m for m in scout_results if m.get("ollama_name")]
        
        best_scout = ollama_results[0]["ollama_name"] if ollama_results else "gemma2:2b"
        
        self.app.suggested_model = suggested
        self.app.scout_model = best_scout
        
        self.query_one("#opt-auto", RadioButton).label = ONBOARDING["model"]["opt_auto"].format(model=suggested)
        self.query_one("#opt-scout", RadioButton).label = ONBOARDING["model"]["opt_scout"].format(model=best_scout)
        
        self.query_one("#confirm", Button).disabled = False

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        is_custom = event.pressed.id == "opt-custom"
        self.query_one("#custom-container", Vertical).set_class(not is_custom, "hidden")
        if is_custom:
            self.query_one("#custom-model-input", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            selected_id = self.query_one("#model-options", RadioSet).pressed_button.id
            if selected_id == "opt-auto":
                self.app.selected_model = self.app.suggested_model
            elif selected_id == "opt-scout":
                self.app.selected_model = self.app.scout_model
            else:
                self.app.selected_model = self.query_one("#custom-model-input", Input).value or "gemma2:2b"
            
            self.app.push_screen(VoiceInterviewScreen())

class VoiceInterviewScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Container(
            Label(ONBOARDING["voice"]["title"], id="voice-title"),
            Static(ONBOARDING["voice"]["desc"], id="voice-desc"),
            
            Label(ONBOARDING["voice"]["name_label"]),
            Input(placeholder=ONBOARDING["voice"]["name_placeholder"], id="user-name-input"),
            
            Horizontal(
                Label(ONBOARDING["voice"]["voice_toggle_label"]),
                Switch(value=True, id="voice-toggle"),
                id="voice-switch-container"
            ),
            
            Horizontal(
                Button(ONBOARDING["voice"]["finish_btn"], id="finish", variant="success"),
                id="button-container"
            ),
            id="voice-container"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "finish":
            user_name = self.query_one("#user-name-input", Input).value or "User"
            voice_enabled = self.query_one("#voice-toggle", Switch).value
            
            self.app.results = {
                "engine": getattr(self.app, "engine_type", "ollama"),
                "user_name": user_name,
                "voice_enabled": voice_enabled,
                "hardware_profile": getattr(self.app, "hardware_profile", None)
            }
            
            if self.app.results["engine"] == "api":
                self.app.results.update(self.app.api_results)
            else:
                self.app.results.update({
                    "model": getattr(self.app, "selected_model", "gemma2:2b"),
                    "provider": "ollama",
                    "base_url": "http://localhost:11434",
                    "embedding_model": "nomic-embed-text:latest"
                })
            
            self.app.exit(self.app.results)

class OnboardingApp(App[dict[str, Any]]):
    TITLE = ONBOARDING["app_title"]
    CSS = """
    Screen {
        align: center middle;
    }

    #welcome-container, #hardware-container, #model-container, #voice-container, #backend-container, #api-container {
        width: 70;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }

    #ascii-art {
        color: $accent;
        width: 100%;
        content-align: center middle;
        margin-bottom: 1;
    }

    #welcome-title, #scan-label, #model-title, #voice-title, #backend-title, #api-title {
        text-style: bold;
        width: 100%;
        content-align: center middle;
        margin-bottom: 1;
    }

    #welcome-text, #model-desc, #voice-desc, #backend-desc {
        margin-bottom: 2;
        text-align: center;
    }

    #hardware-panel {
        background: $panel;
        padding: 1;
        margin-bottom: 1;
        border: solid $primary-darken-2;
    }

    #button-container {
        margin-top: 2;
        height: auto;
        align: center middle;
    }

    .hidden {
        display: none;
    }

    #model-options {
        margin-bottom: 1;
    }

    #custom-container {
        margin-top: 1;
        background: $panel;
        padding: 1;
    }

    #voice-switch-container {
        height: auto;
        margin-top: 1;
        align: left middle;
    }

    #voice-switch-container Label {
        margin-right: 2;
    }
    """

    def on_mount(self) -> None:
        self.push_screen(WelcomeScreen())

if __name__ == "__main__":
    app = OnboardingApp()
    print(app.run())
