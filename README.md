# 🎙️ FRIDAY: Local-First AI Assistant

Friday is a privacy-centric, offline-first personal assistant. It combines high-speed local inference (Ollama) with high-quality voice (Piper/Vosk) and a secure code execution sandbox.

## 🏗️ Architecture
- **Core**: Registry-based command dispatcher with session persistence.
- **IO**: Rich-powered CLI with dual Text/Voice input modes.
- **Security**: Subprocess sandboxing with `unshare` and resource `ulimits`.
- **Voice**: 100% local TTS (Piper) and STT (Vosk).

## 🚀 Quick Start
1. **Install Dependencies**:
   ```bash
   pip install -e .
   ```
2. **Setup Models**:
   ```bash
   friday voice download
   ```
3. **Verify Health**:
   ```bash
   friday doctor
   ```
4. **Launch**:
   ```bash
   friday
   ```

## 🛠️ Configuration
Edit `~/.friday/config.yaml` to change LLM models, voice speed, or sandbox timeouts.
