# 🎙️ FRIDAY v2: Self-Improving Multi-Platform AI Agent

**FRIDAY** (Female Replacement Intelligent Digital Assistant Youth) has evolved. Version 2 is a production-ready, self-improving personal AI agent that operates across multiple platforms and grows its own capabilities.

---

## ✨ v2 Key Features
- **Multi-Platform Gateways**: Seamlessly interact via Terminal (TUI), Telegram, Discord, or Slack.
- **Self-Improving Skill System**: FRIDAY can dynamically generate, execute, and repair Python skills to handle new tasks.
- **Advanced Memory Architecture**: 
  - **Structured Memory**: SQLite-based conversation logs.
  - **Semantic Memory**: ChromaDB vector store for long-term recall and contextual awareness.
- **Flexible LLM Providers**: Support for OpenRouter (Gemini, Claude, Llama 3), OpenAI, and Ollama (Local).
- **Task Scheduler**: Built-in cron-based scheduling for automated reports and tasks.
- **Modern Tech Stack**: Python 3.11+, Pydantic v2, Asyncio, and Rich TUI.

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/your-repo/friday.git
cd friday
pip install -r requirements.txt
```

### 2. Configuration
Copy the example environment file and fill in your API keys:
```bash
cp .env.example .env
```
Edit `~/.friday/config.yaml` to configure your preferred LLM provider and enable gateways.

### 3. Launching FRIDAY
**Start the Interactive TUI:**
```bash
python -m friday.main chat
```

**Start the Gateways (Telegram, Discord, etc.):**
```bash
python -m friday.main gateway --telegram --discord
```

---

## 🛠️ CLI Usage
FRIDAY comes with a powerful CLI for one-off tasks:
- `friday chat`: Launch the terminal UI.
- `friday ask "question"`: Quick query to the agent.
- `friday gateway start --telegram`: Launch specific platform gateways.
- `friday skill list`: View all current capabilities.
- `friday schedule add "Daily Report" --cron "0 9 * * *" --prompt "Summarize my day"`: Schedule recurring tasks.

---

## 🧠 The Self-Improvement Loop
When FRIDAY encounters a task she doesn't have a skill for, she can:
1. **Design**: Write a new Python skill in `friday/skills/`.
2. **Implement**: Register the skill in the index.
3. **Execute**: Run the code immediately to fulfill the request.
4. **Repair**: If the code fails, she analyzes the error and regenerates the skill.

---

## ⚙️ Configuration Example (`config.yaml`)
```yaml
llm:
  provider: openrouter
  model: google/gemini-2.5-flash-preview
  api_key: ${OPENROUTER_API_KEY}

gateways:
  telegram:
    enabled: true
    token: ${TELEGRAM_BOT_TOKEN}
```

---

## 🛡️ Security & Integrity
FRIDAY isolates skill execution and uses environment-based configuration to keep your secrets safe. Always review the skills FRIDAY creates in the `friday/skills/` directory.
