# FRIDAY User Manual: A Complete Guide

Welcome! This guide is designed to help anyone—regardless of technical skill—get the most out of **FRIDAY**, your personal AI assistant.

---

## 🏁 Setting Up for the First Time

### 1. The Setup Assistant (`friday init`)
The easiest way to configure FRIDAY is to use her built-in setup assistant. Open your terminal and type:
```bash
friday init
```
This tool will:
- Check how powerful your computer is (CPU, RAM, GPU).
- Recommend the best AI models for your specific machine.
- Let you choose between a **Male** or **Female** voice.
- Automatically download the necessary voice files.

### 2. Ensuring the "Brain" is Ready
FRIDAY uses **Ollama** to think. 
- Make sure Ollama is running in the background (you should see its icon in your taskbar or system tray).
- If FRIDAY says she can't find a model, run: `ollama pull mistral` (or the model name recommended during `init`).

---

## 💬 Interacting with FRIDAY

### Starting the Chat
Simply type `friday` in your terminal. You'll see a `>>>` prompt when she is ready for you to type.

### Talking and Listening (Voice Mode)
- **Turn it on**: Type `/voice on`. FRIDAY will now listen to your microphone.
- **How to speak**: Speak clearly after you see the `Listening...` message.
- **Turn it off**: Type `/voice off` to return to typing only.
- **Speaker only**: If you want her to speak but you want to keep typing, run `friday -v`.

### Asking Quick Questions
You don't have to enter the full chat mode for a single question. Use the `ask` command:
```bash
friday ask "What is the distance to the moon?"
```

---

## 🧠 Features & Capabilities

| Feature | What it does | Example Command |
| :--- | :--- | :--- |
| **Identity** | Tells you about her purpose. | `Who are you?` |
| **Files** | Creates or reads files for you. | `Create a shopping list for a BBQ` |
| **Coding** | Writes simple scripts or code. | `Write a Python script to say hello` |
| **Memory** | Remembers the current chat. | `My name is Alex` -> `What is my name?` |
| **Forget** | Clears the current short-term memory. | `Clear history` |
| **Digest** | Summarizes news or your schedule. | `Morning digest` |

---

## 🛠️ Troubleshooting (FAQ)

### ❓ "Friday is not responding" or "Connection Error"
- **Cause**: Ollama is likely not running.
- **Fix**: Open the Ollama application on your computer. If it's already open, try restarting it.

### ❓ "I can't hear anything"
- **Cause**: Voice models might be missing or volume is low.
- **Fix 1**: Run `friday voice download` to ensure all voice files are installed.
- **Fix 2**: Check your system volume and ensure the correct output device (speakers/headphones) is selected.

### ❓ "She can't hear me" (Microphone issues)
- **Cause**: Incorrect microphone selection or missing permissions.
- **Fix 1**: Check your system settings to ensure your microphone is the "Default Input Device."
- **Fix 2**: Run `friday doctor` to see if she detects any microphones.

### ❓ Linux Users: "PyAudio" or "PortAudio" errors
- **Cause**: Linux requires an extra library for sound.
- **Fix**: Run this command in your terminal:
  ```bash
  sudo apt-get install portaudio19-dev
  ```

### ❓ Running a Health Check
If you are unsure what's wrong, run:
```bash
friday doctor
```
This will check your connection to Ollama, verify your voice models, and test your microphone.

---

## 🔗 Connecting Your Accounts (Optional)

FRIDAY can become even more helpful if you connect her to your existing services. This allows features like the **Morning Digest** to show your real data instead of "mock" examples.

To do this, you can create a file named `.env` in the same folder as FRIDAY and add your details:

### 📧 Email (IMAP)
Add these lines to your `.env` file:
```text
IMAP_SERVER=imap.gmail.com
IMAP_USER=your-email@gmail.com
IMAP_PASSWORD=your-app-password
```
*(Note: For Gmail, you'll need to use an "App Password" rather than your main password).*

### 🗞️ News
FRIDAY can fetch the latest headlines if you provide a news API key:
```text
NEWS_API_KEY=your_key_here
```

---

## 🛡️ Privacy & Safety
- **Offline First**: FRIDAY does not upload your voice or text to any servers. 
- **Sandbox**: When you ask FRIDAY to write or run code, she does it in a "Sandbox." This is a restricted environment that prevents the code from accessing your private files or damaging your system.
