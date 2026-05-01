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
Run the automatic setup assistant. It uses our advanced **Model Scout** technology to check your hardware and recommend the best local LLMs for your specific machine:
```bash
friday init
```
- Select your **LLM Backend** (Use `1` for the default Local Ollama).
- Review the hardware-tailored model recommendations.
- Choose a **Male or Female** voice.
- Let it download and pull everything automatically.

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
- **JSON Output**: For scripting, run `friday model-scout --json`.

### 📋 Common Commands & Tasks
| What to Ask | What FRIDAY Does |
| :--- | :--- |
| **"Who are you?"** | Introduces herself and her purpose. |
| **"Create a file named notes.txt"** | Writes a new file to your computer safely. |
| **"Morning digest"** | Gives you a summary of news or your schedule. |
| **"Clear history"** | Makes her forget the current conversation. |
| **"Write a Python script to..."** | Writes code for you and runs it in her "Safe Zone." |
| **"model-scout"** | Opens the hardware compatibility assessment tool. |

---

## 🛠️ Troubleshooting (Help!)

If something isn't working right, don't worry! Here are the most common fixes:

### ❓ FRIDAY is not responding or says "Connection Error"
- **The Fix**: Make sure **Ollama** is running. Check your taskbar (Windows) or Menu Bar (Mac) for the Ollama icon. If it's closed, open it and try again.

### ❓ "I can't hear anything"
- **The Fix 1**: Run `friday voice download` to make sure the voice files are installed.
- **The Fix 2**: Check your computer's volume and ensure your speakers are turned on.

### ❓ "She can't hear me" (Microphone issues)
- **The Fix 1**: Make sure your microphone is plugged in and set as the "Default" in your computer's sound settings.
- **The Fix 2**: Run `friday doctor` to see if FRIDAY can find your microphone.

### ❓ Linux Users: Sound errors
- **The Fix**: If you see errors about "PyAudio" or "PortAudio," run this command:
  ```bash
  sudo apt-get install portaudio19-dev
  ```

### ❓ Still having trouble?
Run the "Doctor" command for a full system health check:
```bash
friday doctor
```

---

## 🏗️ Technical Highlights (For the Curious)

- **🛡️ Hardened Sandboxing**: All code execution is isolated in restricted subprocesses or Docker containers.
- **🚀 Optimized Memory**: Uses high-performance batch embedding to index your local documents up to 10x faster.
- **🔄 Resilient Routing**: Automatically switches to secondary models if your primary choice is unavailable.
- **🔒 Thread-Safe**: Background tasks are synchronized to maintain perfect conversation context.

---

## 📧 Optional: Connecting Accounts

To make FRIDAY even smarter (like reading your real emails for a Morning Digest), you can add your details to a `.env` file in the project folder:

```text
IMAP_SERVER=imap.gmail.com
IMAP_USER=your-email@gmail.com
IMAP_PASSWORD=your-app-password
```

For more support, please check the GitHub issues page.
