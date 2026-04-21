# FRIDAY User Manual

## 🎮 Basic Interaction
- **Text Mode**: Type commands directly into the `>>>` prompt.
- **Voice Mode**: Type `voice on` to enable the microphone. Friday will listen for silence before processing. Type `voice off` to return to text.
- **Exit**: Type `exit`, `quit`, or `bye`.

## 🤖 Available Commands
- `help`: Lists all available tools.
- `who are you?`: Identity and system info.
- `clear history`: Resets the AI's short-term memory.
- `create a file [name]`: Triggers the Code Assistant to write and execute Python code.

## ⚙️ Administrative Commands
- `friday voice download`: Fetches the Jenny (Female) TTS model and Vosk STT models.
- `friday doctor`: Runs a diagnostic check on your local Ollama instance and audio drivers.

## 🔒 Security Policy
Friday executes code in a restricted sandbox. On Linux, it attempts to use `unshare` to disable network access during execution. Files created by Friday are stored in `~/Desktop` or `~/.friday/workspace` by default.
