# 🎙️ FRIDAY: Your Personal Offline AI Assistant

**FRIDAY** (Female Replacement Intelligent Digital Assistant Youth) is a privacy-first, local AI assistant that lives on your computer. Unlike other assistants, FRIDAY doesn't send your data to the cloud—everything stays on your machine.

---

## ✨ Why choose FRIDAY?
- **🔒 100% Private**: Your conversations and data never leave your computer.
- **🗣️ Talk & Listen**: Fully voice-enabled interaction.
- **🧠 Local Memory**: She remembers your past chats and can "read" your local documents.
- **⚡ Fast & Free**: No subscriptions, no internet required (after setup).
- **🛡️ Safe Coding**: Runs scripts and code in a secure "Sandbox" to protect your files.

---

## 🚀 Getting Started (Easy Setup)

Follow these steps to bring FRIDAY to life on your computer.

### 1. Prerequisites
Before installing FRIDAY, you need:
1. **Python (3.10 or newer)**: Download from [python.org](https://www.python.org/downloads/).
2. **Ollama**: This is the "brain" that runs the AI. Download from [ollama.com](https://ollama.com).
   - Once installed, open your terminal/command prompt and run:
     ```bash
     ollama pull mistral
     ```
3. **Docker (Optional but Recommended)**: Required for maximum security when FRIDAY runs code. Download from [docker.com](https://www.docker.com/).

### 2. Installation
Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run these commands:

```bash
# Clone this repository (or download and unzip it)
git clone https://github.com/your-repo/friday.git
cd friday

# Install FRIDAY
pip install -e .
```

### 3. First-Time Setup
Run the automatic setup assistant. It will check your hardware and help you download everything you need:
```bash
friday init
```

---

## 💬 How to Use FRIDAY

### ⌨️ The Interactive Chat
Type `friday` to start a conversation.
- Type **`/voice on`** to start talking with your microphone.
- Type **`/voice off`** to go back to typing.
- Type **`/exit`** to say goodbye.

### ❓ Quick Questions
Ask a quick question without entering full chat mode:
```bash
friday ask "How do I make a perfect omelette?"
```
*Add `-v` at the end to hear her speak the answer!*

### 📋 Common Commands
- **"Who are you?"** - FRIDAY introduces herself.
- **"Create a file named notes.txt"** - She creates a file for you.
- **"Morning digest"** - Get your daily summary (if configured).
- **"Clear history"** - Make her forget the current conversation.

---

## 🏗️ Technical Highlights

FRIDAY is built with modern engineering standards to ensure speed, safety, and reliability:

- **🛡️ Hardened Sandboxing**: All code execution is isolated in restricted subprocesses or Docker containers. Every script undergoes static analysis to prevent unauthorized imports or dangerous system calls.
- **🚀 Optimized Vector Memory**: Uses high-performance batch embedding to index your local documents up to 10x faster than traditional sequential methods.
- **🔄 Resilient LLM Routing**: Features an intelligent failover system that automatically switches to secondary models if your primary choice is unavailable, and gracefully "upgrades" back once service is restored.
- **🔒 Thread-Safe Semantic Memory**: Background summarization tasks are synchronized with async locks to maintain perfect conversation context, even during rapid-fire interactions.

---

## 🛠️ System Health
If something isn't working right, run the "Doctor" command:
```bash
friday doctor
```

For more detailed instructions, check out the [USER_MANUAL.md](USER_MANUAL.md).
