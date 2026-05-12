# 🎙️ FRIDAY: Your Personal Offline AI Assistant

**A local-first, privacy-centric framework for digital empowerment.**

FRIDAY (Female Replacement Intelligent Digital Assistant Youth) is an open-source framework designed to bring high-performance AI agents directly to your local machine. Unlike cloud-based assistants, FRIDAY ensures your data never leaves your computer while providing a rich, voice-enabled experience.

---

## 📖 Table of Contents
- [Description](#description)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration & Environment](#configuration--environment)
- [Features](#features)
- [API Documentation (Extending Friday)](#api-documentation-extending-friday)
- [Contributors](#contributors)
- [License](#license)

---

## 📝 Description
FRIDAY is built for those who value privacy without compromising on AI capabilities. It serves as a personal assistant capable of chatting, answering questions, managing your morning routine, and even executing code in a safe environment. By leveraging local LLM engines like **Ollama**, FRIDAY provides a "Safe Zone" for digital interaction.

---

## 🛠️ Prerequisites
Before bringing FRIDAY to life, ensure your system meets these requirements:

1.  **Python (3.10 or newer)**: The core engine. Download from [python.org](https://www.python.org/downloads/).
2.  **Ollama**: The "brain" for local LLMs. Download from [ollama.com](https://ollama.com).
3.  **Docker (Optional)**: Highly recommended for the **Code Assistant**. It creates a "sealed box" (sandbox) to run generated code safely.
4.  **PortAudio (Linux only)**: Required for voice features.
    ```bash
    sudo apt-get install portaudio19-dev
    ```

---

## 🚀 Installation
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repo/friday.git
    cd friday
    ```

2.  **Install the Package**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -e .
    ```

---

## 💬 Usage
FRIDAY provides several ways to interact with her.

### 1. First-Time Setup
Run the interactive onboarding TUI to configure your backend and scan your hardware:
```bash
friday init
```

### 2. Interactive Chat Mode
Enter a full conversational loop:
```bash
friday
```
- **Voice Mode**: Type `/voice on` to enable the microphone.
- **Exit**: Type `/exit` or `/quit`.

### 3. Quick Questions (One-Shot)
Ask a single question and exit:
```bash
friday ask "How do I make a perfect omelette?"
```
*Tip: Add `-v` (e.g., `friday ask "Hello" -v`) to hear her speak the answer.*

### 4. Hardware Assessment (Model Scout)
Find the best models for your specific CPU, RAM, and GPU:
```bash
friday model-scout
```

### 5. Health Check
Run a diagnostic to ensure all systems (STT, TTS, APIs) are working:
```bash
friday doctor
```

---

## ⚙️ Configuration & Environment
FRIDAY stores her settings in your home directory: `~/.friday/config.yaml`.

### Common Commands:
- **View Config**: `friday config list`
- **Update Setting**: `friday config set llm.primary_model "llama3:8b"`

### Environment Overrides:
You can also set environment variables to override config values:
- `FRIDAY_LLM_PRIMARY_MODEL`
- `FRIDAY_LLM_API_KEY`

---

## ✨ Features
- **🌍 Multi-Provider LLM**: Support for Ollama, OpenAI, Gemini, Mistral, Groq, and OpenRouter.
- **🗣️ Full Voice Integration**: High-quality local Text-to-Speech (TTS) and Speech-to-Text (STT).
- **🧠 Vector Memory**: Remembers past conversations and indexes local documents for context.
- **🛡️ Secure Sandbox**: Executes Python code and system commands in a Docker-protected environment.
- **🔍 Model Scout**: Intelligent hardware scanning to recommend the best local models for your machine.
- **🔌 MCP Support**: Native support for the Model Context Protocol to connect external tools.

---

## 🔌 API Documentation (Extending Friday)
FRIDAY is designed to be a framework. You can extend her by adding new **Agents** or **Plugins**.

### Creating a Custom Agent
All agents must inherit from the `BaseAgent` class found in `src/friday/agents/base.py`:
```python
from friday.agents.base import BaseAgent, Context, AgentResult

class MyAgent(BaseAgent):
    async def run(self, ctx: Context) -> AgentResult:
        # Your logic here
        return AgentResult(content="Hello from my custom agent!")

    @property
    def name(self) -> str:
        return "my_custom_agent"
```

### Adding a Plugin
Create a new directory in `src/friday/plugins/` with a `manifest.json`:
```json
{
  "name": "my_plugin",
  "version": "1.0.0",
  "description": "A custom plugin for Friday.",
  "entry_point": "friday.plugins.my_plugin.main"
}
```

---

## 👥 Contributors
- **Raj** (Lead Developer) - [raj@example.com](mailto:raj@example.com)

---

## 📄 License
This project is licensed under the **MIT License**. See the `LICENSE` file (if available) or `pyproject.toml` for details.
