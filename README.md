# FRIDAY

A local-first, privacy-centric personal AI assistant framework.

## Core Philosophy
- **Local-First by Default**: All inference, embedding, and TTS must default to local models.
- **Privacy Boundary**: No telemetry, no external logging. User data never leaves the machine.
- **Hardware Adaptive**: Auto-detect hardware and select appropriate model quantization and inference backend.

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

## Development
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```
