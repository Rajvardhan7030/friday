# 🎙️ FRIDAY: Your Personal Offline AI Assistant

**FRIDAY** (Female Replacement Intelligent Digital Assistant Youth) is a privacy-first, local AI assistant that lives on your computer. Unlike other assistants, FRIDAY doesn't send your data to the cloud—everything stays on your machine.

---

## ✨ Key Features
- **Privacy First**: 100% local processing. No data leaves your computer.
- **Voice Interaction**: Talk to FRIDAY and hear her speak back.
- **Memory**: FRIDAY remembers your previous conversations and can even "read" your local documents to help you better.
- **Code Assistant**: Ask FRIDAY to create files or run small scripts for you.
- **Daily Briefings**: Get a "Morning Digest" of your news and schedule (if configured).

---

## 🚀 Easy Setup (For Everyone)

### 1. Install Ollama (The "Brain")
FRIDAY needs a local brain to think. We recommend **Ollama**:
1. Download Ollama from [ollama.com](https://ollama.com).
2. Install it and run it.
3. Open your terminal and type:
   ```bash
   ollama pull mistral
   ```

### 2. Install FRIDAY
Open your terminal and run:
```bash
pip install -e .
```

### 3. Download Voice Models
To let FRIDAY speak and listen, run this command:
```bash
friday voice download
```

### 4. Check if everything is ready
Run the "doctor" command to make sure FRIDAY is healthy:
```bash
friday doctor
```

---

## 💬 How to Use FRIDAY

### The Interactive Chat
Simply type `friday` in your terminal to start a conversation.
- Type `help` to see what she can do.
- Type `/voice on` to start talking to her with your microphone.
- Type `/voice off` to go back to typing.
- Type `/exit` when you're done.

### One-Off Questions
If you just want a quick answer without staying in the chat:
```bash
friday ask "What is the capital of France?"
```
To hear the answer out loud:
```bash
friday ask -v "Tell me a joke"
```

---

## 🛠️ Common Tasks
- **Greeting**: "Hi Friday!"
- **Identity**: "Who are you?"
- **Coding**: "Create a file named hello.py that prints 'Hello World'"
- **Daily Briefing**: "Give me my morning digest"
- **Clear Memory**: "Clear history"

---

## 🔒 Your Privacy & Security
FRIDAY is designed to be safe. When you ask her to write code or create files, she does so in a "Sandbox"—a restricted area of your computer that prevents the code from accessing your network or private data without your knowledge.

## ⚙️ Advanced Configuration
Technical users can find settings in `~/.friday/config.yaml` to change models, voice speed, or folder locations.
