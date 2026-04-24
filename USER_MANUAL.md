# FRIDAY User Manual

## 🎮 Basic Interaction
- **Interactive Shell**: Run `friday` to open the main REPL.
- **One-Shot Ask**: Run `friday ask "question"` to get one answer and exit.
- **Spoken One-Shot Ask**: Run `friday ask -v "question"` to hear the answer aloud.
- **Text Mode**: In the interactive shell, type normal requests directly into the `>>>` prompt.
- **Voice Mode**: Type `/voice on` to enable the microphone and spoken replies. Type `/voice off` to return to text-only mode.
- **Exit**: Type `/exit`, `/quit`, or `/bye`.

## 🧭 Command Rules
- Slash-prefixed commands are reserved for shell controls such as `/voice on` and `/exit`.
- Regular assistant requests should stay as plain text, such as `help` or `who are you?`.
- `-v` and `--voice-output` enable spoken output for command-line flows like `friday ask`.

## 🤖 Available Commands
- `help`: Lists all available tools.
- `who are you?`: Identity and system info.
- `clear history`: Resets the AI's short-term memory.
- `create a file [name]`: Triggers the Code Assistant to write and execute Python code.

## ⚙️ Administrative Commands
- `friday voice download`: Fetches the Jenny (Female) TTS model and Vosk STT models.
- `friday doctor`: Runs a diagnostic check on your local Ollama instance and audio drivers.
- `friday ask "question"`: Runs a one-shot prompt without opening the interactive shell.
- `friday ask -v "question"`: Runs a one-shot prompt and speaks the response aloud.

## 📝 Typical Examples
- `friday`
- `friday ask "summarize my last task"`
- `friday ask -v "what time is it"`
- `/voice on`
- `/voice off`
- `/exit`

## 🔒 Security Policy
Friday executes code in a restricted sandbox. On Linux, it attempts to use `unshare` to disable network access during execution. Files created by Friday are stored in `~/Desktop` or `~/.friday/workspace` by default.
