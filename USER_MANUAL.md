# FRIDAY User Manual: A Simple Guide

Welcome! This guide will help you get the most out of **FRIDAY**, your personal AI assistant. No technical degree required!

---

## 🏁 Getting Started

### 1. How to talk to FRIDAY
Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and type:
`friday`

You will see a prompt like this: `>>>`. This is where you type your questions!

### 2. Voice Mode (Talking and Listening)
If you'd rather talk than type:
1. Type `/voice on` in the chat and press Enter.
2. Wait for FRIDAY to say "Voice mode enabled."
3. Speak clearly into your microphone. FRIDAY will listen, think, and speak back to you!
4. To stop talking and go back to typing, type `/voice off`.

---

## 🤖 What can I ask FRIDAY?

FRIDAY can help with many things. Here are some examples of what you can type:

| What you want | What to type/say |
| :--- | :--- |
| **Get Help** | `help` |
| **Friendly Hello** | `Hi Friday!` |
| **Check Identity** | `Who are you?` |
| **Write a file** | `Create a file named shopping_list.txt` |
| **Write a script** | `Write a python script that calculates the area of a circle` |
| **Daily Briefing** | `Morning digest` |
| **Clear History** | `Clear history` |

---

## 🧠 Memory: How FRIDAY remembers you

FRIDAY has two types of memory:
1. **Short-Term**: She remembers the conversation you are having *right now*. If you say "My name is John," and then ask "What is my name?", she will know.
2. **Long-Term**: If enabled, FRIDAY can "read" files in her workspace folder (`~/.friday/workspace`) to help answer your questions about your own documents.

To start fresh and make her forget the current conversation, just type:
`clear history`

---

## ❓ Troubleshooting (When things go wrong)

### "Friday is not responding"
- Make sure **Ollama** is running on your computer.
- Check your internet connection (only needed for the initial setup/downloads).

### "I can't hear Friday"
- Make sure your speakers are turned on and the volume is up.
- Run `friday voice download` to ensure the voice files are installed.

### "Friday can't hear me"
- Check that your microphone is plugged in and selected as the default input in your system settings.
- Type `/voice on` to make sure she is listening.

### Running a Health Check
If you are having trouble, run this command in your terminal:
`friday doctor`
This will tell you if any parts of FRIDAY are "sick" or missing.

---

## 🛡️ Staying Safe
FRIDAY is built with security in mind. When she runs code to help you with a task, she does it in a "Sandbox." This is like a virtual room that she can't leave, so the code can't accidentally delete your important files or spy on your other apps.
