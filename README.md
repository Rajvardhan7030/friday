# 🎙️ FRIDAY: Your Personal Offline AI Assistant

**FRIDAY** (Female Replacement Intelligent Digital Assistant Youth) is a privacy-first, local AI assistant that lives on your computer. Unlike other assistants (like Siri or Alexa), FRIDAY doesn't send your data to the cloud—everything stays on your machine.

---

## ✨ Why choose FRIDAY?
- **🔒 100% Private**: Your conversations and data never leave your computer.
- **🗣️ Talk & Listen**: Fully voice-enabled interaction.
- **🧠 Local Memory**: She remembers your past chats and can "read" your local documents.
- **⚡ Fast & Free**: No subscriptions, no internet required (after setup).
- **🛡️ Safe Zone**: Runs scripts and code in a secure "Sandbox" to protect your files.

---

## 🚀 Getting Started

Follow these steps to bring FRIDAY to life on your computer.

### 1. Simple Prerequisites
Before installing FRIDAY, you need:
1.  **Python (3.10 or newer)**: Think of this as the engine that runs FRIDAY. Download it from [python.org](https://www.python.org/downloads/).
2.  **Ollama**: This is the "brain" that lets FRIDAY think and talk. Download it from [ollama.com](https://ollama.com).
    - *Tip: Once installed, keep it running in your taskbar.*
3.  **Docker (Optional)**: For extra safety when you ask FRIDAY to write code, install [Docker](https://www.docker.com/). It keeps her coding in a "sealed box."

### 2. Installation
Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:

```bash
# Get the code
git clone https://github.com/your-repo/friday.git
cd friday

# Install FRIDAY
pip install -e .
```

### 3. First-Time Setup
Run the interactive setup assistant. It features a modern **TUI (Terminal User Interface)** to help you configure your AI backend and use our **Model Scout** technology to recommend the best models for your specific machine:
```bash
friday init
```
- **Backend Choice**: Select **Local Ollama** for 100% privacy, or high-speed **Cloud APIs** (Gemini, OpenAI, Mistral, Groq, OpenRouter).
- **Auto-Config**: The TUI pre-fills the best models and URLs for your selected provider.
- **Hardware Scan**: Review tailored recommendations based on your CPU, RAM, and GPU (NVIDIA or Apple Silicon).
- **Voice Selection**: Choose a **Male or Female** voice and let FRIDAY pull everything for you.

---

## 💬 How to Use FRIDAY

### ⌨️ Starting a Conversation
Type `friday` to start chatting.
- **Voice Mode**: Type `/voice on` to start talking. Type `/voice off` to go back to typing.
- **Exit**: Type `/exit` or `/quit` when you're done.

### ❓ Quick Questions
Ask a quick question without entering full chat mode:
```bash
friday ask "How do I make a perfect omelette?"
```
*Add `-v` to the end (e.g., `friday ask "Hello" -v`) to hear her speak the answer!*

### 🔍 Model Scout (Hardware Assessment)
Want to know which local models your computer can handle? Use the interactive Scout TUI:
```bash
friday model-scout
```
- **Sort & Filter**: Find models by tag (coding, reasoning) or suitability.
- **Performance Estimates**: See estimated tokens/second for your CPU/GPU.
- **One-Click Setup**: Press `D` to see the pull command for any compatible model.

### 📋 Common Commands & Tasks
| Command | What it does |
| :--- | :--- |
| `friday` | Enters full interactive chat mode. |
| `friday init` | Launches the interactive TUI setup assistant. |
| `friday status` | Shows current LLM provider, models, and memory status. |
| `friday doctor` | Runs a system health check (Hardware, STT, TTS, APIs). |
| `friday model-scout` | Opens the hardware compatibility assessment tool. |
| `friday ask "..."` | One-shot question mode (add `-v` for voice). |
| `friday config` | View or modify configuration settings via CLI. |

---

## 🛠️ Troubleshooting (Help!)

If something isn't working right, don't worry! Here are the most common fixes:

### ❓ Connection Error or "Brain Foggy"
- **Ollama Users**: Make sure **Ollama** is running in your taskbar.
- **API Users**: Run `friday status` to check your provider. Ensure your API Key is correct in `config.yaml`.
- **Gemini Users**: FRIDAY automatically sanitizes your parameters and strips model prefixes to avoid "400 Bad Request" errors.

### ❓ "I can't hear anything" / "She can't hear me"
- Run `friday voice download` to ensure voice models are present.
- Run `friday doctor` to verify your microphone is detected and energy levels are correct.

### ❓ Sound errors on Linux
- **The Fix**: If you see errors about "PyAudio" or "PortAudio," run:
  ```bash
  sudo apt-get install portaudio19-dev
  ```

---

## 🏗️ Technical Highlights

- **🌍 Multi-Provider LLM**: Native support for **Ollama, OpenAI, Google Gemini, Mistral, Groq, and OpenRouter**.
- **🛡️ Robust API Sanitization**: Specialized engines automatically strip prefixes (like `models/`) and sanitize unsupported parameters (like `presence_penalty` or `strict` schemas) before they hit provider endpoints.
- **📱 Apple Silicon Optimized**: Full VRAM detection and Tok/s heuristics for Mac M1/M2/M3 chips.
- **🚀 Vector Memory**: High-performance local document indexing using batch embeddings.

---

## 📧 Optional: Connecting Accounts

To make FRIDAY even smarter (like reading your real emails for a Morning Digest), you can add your details to a `.env` file in the project folder:

```text
IMAP_SERVER=imap.gmail.com
IMAP_USER=your-email@gmail.com
IMAP_PASSWORD=your-app-password
```

For more support, please check the GitHub issues page.
