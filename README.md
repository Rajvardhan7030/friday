# 🎙️ FRIDAY: Local-First AI Assistant

Friday is a privacy-centric, offline-first personal assistant. It combines high-speed local inference (Ollama) with high-quality voice (Piper/Vosk) and a secure code execution sandbox.

## 🏗️ Architecture
- **Core**: Registry-based command dispatcher with session persistence.
- **IO**: Rich-powered CLI with dual Text/Voice input modes.
- **Security**: Subprocess sandboxing with `unshare` and resource `ulimits`.
- **Voice**: 100% local TTS (Piper) and STT (Vosk).

## 🚀 Quick Start
1. **Install Dependencies**:
   ```bash
   pip install -e .
   ```
2. **Setup Models**:
   ```bash
   friday voice download
   ```
3. **Verify Health**:
   ```bash
   friday doctor
   ```
4. **Ask Once**:
   ```bash
   friday ask "who are you?"
   friday ask -v "read this answer aloud"
   ```
5. **Launch**:
   ```bash
   friday
   ```

## 💬 CLI Modes
- `friday`: Starts the interactive shell.
- `friday ask "question"`: Runs a one-shot prompt and exits.
- `friday ask -v "question"`: Runs a one-shot prompt and reads the answer aloud.
- `friday doctor`: Runs a local diagnostics pass.
- `friday voice download`: Downloads the required local voice models.

## 🎛️ Interactive Commands
Inside the interactive shell, slash-prefixed control commands are reserved for the CLI itself:
- `/voice on`: Enables microphone input and spoken responses.
- `/voice off`: Disables microphone input and spoken responses.
- `/exit`, `/quit`, `/bye`: Leaves the interactive shell.

Agent commands remain plain text. Examples:
```text
help
who are you?
clear history
create a file notes.txt
```

## 🔊 Voice Behavior
- `-v` or `--voice-output` enables spoken output for CLI commands like `friday ask`.
- `/voice on` enables live microphone listening inside the interactive shell and also turns on spoken replies.
- `/voice off` returns the shell to text-only mode.

## 🛠️ Configuration
Edit `~/.friday/config.yaml` to change LLM models, voice speed, or sandbox timeouts.
