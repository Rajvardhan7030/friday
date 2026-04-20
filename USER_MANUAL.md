# FRIDAY User Manual

FRIDAY is a local-first, privacy-centric personal AI assistant framework designed to run entirely on your own hardware.

## 1. Prerequisites

Before installing FRIDAY, ensure you have the following installed:
- **Python 3.10 or higher**
- **Ollama**: Required for local LLM inference. Download at [ollama.ai](https://ollama.ai/).
- **FFmpeg**: Required for audio playback (TTS).
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: `choco install ffmpeg`
- **Piper**: The default TTS engine. Ensure the `piper` executable is in your PATH or set `PIPER_PATH`.

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
   This will recommend a model based on your system and set up the default British female voice.

2. **Pull the recommended model**:
   Use Ollama to download the model suggested during initialization:
   ```bash
   ollama pull <model_name>
   ```

3. **Download Friday's Voice**:
   Download the high-quality female voice model:
   ```bash
   friday voice download
   ```

4. **Verify Installation**:
   Check if all components are working correctly:
   ```bash
   friday doctor
   ```

## 4. Usage Guide

### Asking Questions
Interact with FRIDAY using different modes and the optional voice flag:
- **Chat mode (default)**: `friday ask "How are you?"`
- **Voice output**: `friday ask "Tell me a joke" --voice` (or `-v`)
- **Research mode**: Uses multi-hop search (local memory + web) with citations.
  `friday ask "Latest breakthroughs in local LLMs" --mode research`
- **Code mode**: Specialized for programming tasks.
  `friday ask "Write a python script to sort files by extension" --mode code`

*Note: FRIDAY maintains a 10-message conversational memory to allow for natural follow-up questions.*

### Voice Management
Manage Friday's high-quality local TTS:
- **Download voice**: `friday voice download` (fetches the configured model from HuggingFace).
- **Test voice**: `friday voice test "Hello, I am ready to assist you."`

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

### LLM Configuration
You can customize the primary and fallback models:
```yaml
llm:
  primary_model: "qwen2.5-coder:7b"
  fallback_model: "tinyllama"
```

### Voice Configuration
FRIDAY defaults to a British female voice (`en_GB-southern_english_female-low`). You can change this in `config.yaml`:
```yaml
persona_voice: "en_GB-southern_english_female-low"
persona_name: "Friday"
```

## 6. Troubleshooting

### "Ollama Offline" in `friday doctor`
- Ensure the Ollama service is running (`ollama serve`).
- Check if `FRIDAY_LLM_OLLAMA_BASE_URL` is correctly set (default: `http://localhost:11434`).

### "ModuleNotFoundError: No module named 'friday'"
- Ensure you installed the package using `pip install -e .` in the root directory.
- Check if your Python path includes the `src` directory.

### TTS 404 Errors during Download
- Ensure you are using a valid voice name from the [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) repository.
- Use the format: `lang_CODE-voice_name-quality` (e.g., `en_GB-alba-medium`).

### TTS not speaking
- Ensure `ffmpeg` is installed and in your system PATH.
- Verify that `piper` is available or set `PIPER_PATH` environment variable.
- Run `friday voice download` to ensure the voice files are present in `~/.config/friday/voices/`.
