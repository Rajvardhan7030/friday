/# FRIDAY User Manual

FRIDAY is a local-first, privacy-centric personal AI assistant framework designed to run entirely on your own hardware.

## 1. Prerequisites

Before installing FRIDAY, ensure you have the following installed:
- **Python 3.10 or higher**
- **Ollama**: Required for local LLM inference. Download at [ollama.ai](https://ollama.ai/).
- **FFmpeg**: Required for audio playback (TTS).
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: `choco install ffmpeg`

## 2. Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Rajvardhan7030/friday.git
   cd friday
   ```

2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## 3. Initial Setup

1. **Initialize FRIDAY**:
   Run the initialization command to detect your hardware and create a default configuration:
   ```bash
   friday init
   ```
   This will recommend a model based on your system (e.g., `qwen2.5-coder:7b`) and save settings to your config directory.

2. **Pull the recommended model**:
   Use Ollama to download the model suggested during initialization:
   ```bash
   ollama pull <model_name>
   ```

3. **Verify Installation**:
   Check if all components are working correctly:
   ```bash
   friday doctor
   ```

## 4. Usage Guide

### Asking Questions
You can interact with FRIDAY using different modes:
- **Chat mode (default)**: `friday ask "How are you?"`
- **Research mode**: Uses multi-hop search (local memory + web) with citations.
  `friday ask "Latest breakthroughs in local LLMs" --mode research`
- **Code mode**: Specialized for programming tasks.
  `friday ask "Write a python script to sort files by extension" --mode code`

### Morning Digest
Get a briefing of your unread emails, calendar events, and top news headlines:
```bash
friday digest
```
*Note: Requires IMAP and RSS configuration in `~/.config/friday/config.yaml`.*

### Memory Management
Index your local documents to make them searchable by FRIDAY:
- **Index a directory**: `friday memory index --path ~/Documents/ProjectDocs`
- **Clear memory**: `friday memory clear`

### Skill Management
List all built-in and user-contributed skills:
```bash
friday skill list
```

## 5. Advanced Configuration

Settings are stored in `~/.config/friday/config.yaml` (Linux) or equivalent platform directory.

### IMAP (for Email)
Set the following environment variables or add to `config.yaml`:
- `IMAP_SERVER`: Your IMAP server (e.g., imap.gmail.com)
- `IMAP_USER`: Your email address
- `IMAP_PASSWORD`: Your app password

### TTS (for Voice)
FRIDAY uses **Piper** for high-quality local TTS. You can configure the voice in `config.yaml`:
```yaml
persona_voice: "en_GB-vits-low"
```

## 6. Dependencies

FRIDAY relies on the following key libraries:
- **typer**: CLI interface
- **ollama**: LLM communication
- **chromadb**: Vector database for memory
- **pydantic-settings**: Configuration management
- **httpx**: Async HTTP requests
- **pydub**: Audio processing
- **piper-tts**: Local text-to-speech
- **duckduckgo-search**: Privacy-friendly web search
- **psutil**: System monitoring and process management

## 7. Troubleshooting

### "Ollama Offline" in `friday doctor`
- Ensure the Ollama service is running (`ollama serve`).
- Check if `FRIDAY_LLM_OLLAMA_BASE_URL` is correctly set (default: `http://localhost:11434`).

### "ModuleNotFoundError: No module named 'friday'"
- Ensure you installed the package using `pip install -e .` in the root directory.
- Check if your Python path includes the `src` directory.

### TTS not speaking
- Ensure `ffmpeg` is installed and in your system PATH.
- Verify that `piper` is available or set `PIPER_PATH` environment variable.

### Permission Errors in Sandboxed Code
- FRIDAY's sandbox attempts to use `unshare` on Linux for network isolation. If this fails, it falls back to a restricted subprocess. Ensure your user has permissions to run subprocesses.

### Prompt Injection Warnings
- If you see a "Potential prompt injection detected" error, ensure your query does not contain phrases like "ignore previous instructions". This is a security feature to protect the system prompt.
