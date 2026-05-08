"""Narratives and ASCII art for the Friday TUI."""

FRIDAY_ASCII = r"""
 _______  ______  _____  ______   _______ __   __ 
|_______| |_____/   |   |     \ |_______|  \_/   
|       | |    \_ __|__ |_____/ |_______|   |    
                                                 
"""

ONBOARDING = {
    "app_title": "FRIDAY Onboarding",
    "welcome": {
        "title": "Welcome to FRIDAY",
        "text": (
            "Your personal AI framework. Privacy-first, local-first, and built for you.\n\n"
            "Let's get your system configured for optimal performance."
        ),
        "button": "Initiate"
    },
    "backend": {
        "title": "Choose your LLM Backend",
        "text": (
            "Local: Private, no cost, runs on your hardware. Recommended for high-end systems.\n\n"
            "Cloud API: Fast, highly capable, but requires an internet connection and API keys."
        ),
        "local_btn": "Local (Ollama)",
        "cloud_btn": "Cloud API"
    },
    "api": {
        "title": "Cloud API Configuration",
        "provider_label": "Provider Selection",
        "key_label": "API Key",
        "key_placeholder": "Enter your API Key",
        "model_label": "Model Name",
        "model_placeholder": "e.g. gpt-4o",
        "url_label": "Base URL",
        "url_placeholder": "https://...",
        "back_btn": "Back",
        "continue_btn": "Continue"
    },
    "hardware": {
        "scanning": "Scanning System Hardware...",
        "success": "Hardware Scan Complete",
        "failure": "Hardware Scan Failed (Using Defaults)",
        "continue_btn": "Continue"
    },
    "model": {
        "title": "Model Configuration",
        "desc": "Select the best LLM brain for your hardware:",
        "opt_auto": "Hardware Auto-tune (Recommended): [cyan]{model}[/cyan]",
        "opt_scout": "Model Scout High-Performance: [cyan]{model}[/cyan]",
        "opt_custom": "Custom Selection",
        "custom_label": "Enter Ollama Model Name:",
        "custom_placeholder": "e.g. llama3:8b",
        "confirm_btn": "Confirm and Download"
    },
    "voice": {
        "title": "Personalization Phase",
        "desc": "Tell us a bit about yourself and how you'd like to interact.",
        "name_label": "What should I call you?",
        "name_placeholder": "Your Name",
        "voice_toggle_label": "Enable Voice Output",
        "finish_btn": "Finish Setup"
    }
}
