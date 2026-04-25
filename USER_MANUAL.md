# FRIDAY v2 User Manual: The Agent's Guide

Welcome to **FRIDAY v2**, your self-improving AI companion. This manual covers everything from basic chat to managing her new autonomous capabilities.

---

## 🏁 Interface Modes

### 1. Terminal UI (TUI)
Run `python -m friday.main chat` for the full experience.
- **Left Pane**: Your conversation history.
- **Right Pane**: System status and memory recall insights.
- **Input**: Type naturally at the bottom.

### 2. Messaging Platforms
If configured, FRIDAY can live in your Telegram, Discord, or Slack.
- She responds to direct messages and mentions.
- **Context Persistence**: FRIDAY remembers you across platforms thanks to her centralized memory system.

---

## 🤖 Skills: Teaching FRIDAY New Tricks

FRIDAY is no longer limited to built-in commands. She can write her own code.

### Using Existing Skills
Type `/skills` or ask "What can you do?" to see a list of registered skills. To use one, just ask:
- *"Check the weather in London"* (Uses weather skill)
- *"Send an email to Boss"* (Uses email skill)

### Automated Skill Creation
If you ask for something complex, FRIDAY might say:
*"I don't have a skill for that yet. Let me create one..."*
She will write a Python script, save it, and execute it. You can see these in the `friday/skills/` folder.

---

## 🧠 Memory & Context

FRIDAY has a "Semantic Brain" (ChromaDB).
- **Recalling the Past**: She automatically recalls relevant snippets from past conversations to provide better answers.
- **Cross-Platform**: If you tell her something on Telegram, she will remember it when you talk to her on Discord.

---

## 📅 Scheduling Tasks

You can automate FRIDAY using the built-in scheduler:
```bash
friday schedule add "MorningBrief" --cron "0 8 * * *" --prompt "What is on my calendar today?"
```
- This will run every day at 8:00 AM.
- The result will be sent to your primary gateway.

---

## ⚙️ Troubleshooting

### LLM Connectivity
If FRIDAY isn't responding, check your `.env` file and `~/.friday/config.yaml`. 
- Ensure your `OPENROUTER_API_KEY` or `OPENAI_API_KEY` is valid.
- If using **Ollama**, ensure the service is running locally.

### Gateway Issues
If a bot is offline:
- Verify the bot tokens in your config.
- Run `python -m friday.main gateway --<platform>` and check the terminal logs for errors.

### Skill Failures
If a skill crashes:
- FRIDAY will try to fix it herself.
- If she fails, you can manually delete the file in `friday/skills/` and ask her to try again.

---

## 🛡️ Safety Note
FRIDAY v2 can execute Python code she writes. While she tries to be safe, always monitor her activity if you ask her to perform high-risk system operations.
