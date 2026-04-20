# FRIDAY

A local-first, privacy-centric personal AI assistant framework.

## Core Philosophy
- **Local-First by Default**: All inference, embedding, and TTS must default to local models.
- **Privacy Boundary**: No telemetry, no external logging. User data never leaves the machine.
- **Hardware Adaptive**: Auto-detect hardware and select appropriate model quantization and inference backend.
- **Natural Interaction**: High-quality local voice and memory-driven conversations.

## Quick Start
1. Install [Ollama](https://ollama.ai/)
2. Install FRIDAY:
   ```bash
   pip install .
   ```
3. Initialize:
   ```bash
   friday init
   ```
4. Setup Voice (High-quality British Female):
   ```bash
   friday voice download
   ```
5. Start Chatting:
   ```bash
   friday ask "Hello Friday, how are you today?" --voice
   ```

## Key Features
- **Conversational Memory**: Remembers your recent interactions for more natural follow-ups.
- **High-Quality TTS**: Features a refined British female voice using Piper.
- **Multi-Modal Agents**: Specialist agents for research, coding, and general assistance.
- **Local RAG**: Index your own documents to build a private knowledge base.

## Development
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```
