"""Unit tests verifying fixes for identified bugs and broken logic."""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from friday.core.config import Config
from friday.core.hardware import HardwareProfile, get_recommended_model
from friday.core.agent_runner import ToolExecutor
from friday.utils.security import validate_shell_command, run_sandboxed_code
from friday.voice.stt import STTEngine
from friday.memory.vector_store import VectorStore
from friday_model_scout.cli import run_scout


@pytest.mark.asyncio
async def test_config_save_truncation():
    """Verify that saving configuration truncates existing longer content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = Path(tmpdir) / "config.yaml"
        # Write 100KB initial file
        config_file.write_text("extra_key: " + "a" * 100000 + "\n")
        initial_size = config_file.stat().st_size
        
        cfg = Config(config_path=config_file)
        # Reset data to minimal dictionary and save
        cfg._data = {"llm": {"engine": "ollama"}}
        cfg.save()
        
        # Verify content was truncated cleanly and size is dramatically smaller
        final_size = config_file.stat().st_size
        assert final_size < 500
        assert final_size < initial_size
        reloaded = Config(config_path=config_file)
        assert reloaded.get("llm.engine") == "ollama"


@pytest.mark.asyncio
async def test_tool_executor_unbound_args_fix():
    """Verify ToolExecutor handles invalid JSON arguments without UnboundLocalError."""
    mcp_mock = AsyncMock()
    executor = ToolExecutor(mcp_client=mcp_mock)
    
    # Invalid JSON string argument
    tool_calls = [{"function": {"name": "test_tool", "arguments": "{invalid_json"}, "id": "call_1"}]
    results = await executor.execute_tool_calls(tool_calls)
    
    assert len(results) == 1
    assert "Error executing tool 'test_tool'" in results[0].content


def test_hardware_recommendations_exact_threshold():
    """Verify get_recommended_model matches exact threshold specs (12.0 VRAM, 8.0 RAM)."""
    gpu_profile = HardwareProfile(
        os="Linux", cpu_arch="x86_64", cpu_cores=8, cpu_threads=16,
        ram_gb=16.0, gpu_vram_gb=12.0, gpu_name="RTX 3060"
    )
    assert get_recommended_model(gpu_profile) == "llama3:70b"

    ram_profile = HardwareProfile(
        os="Linux", cpu_arch="x86_64", cpu_cores=8, cpu_threads=16,
        ram_gb=8.0, gpu_vram_gb=None, gpu_name=None
    )
    assert get_recommended_model(ram_profile) == "phi3:mini"


def test_validate_shell_command_sudo_flags():
    """Verify validate_shell_command handles sudo with flags correctly."""
    cfg = Config()
    cfg.set("security.shell_command_allow_sudo", True, save=False)
    
    is_safe, reason = validate_shell_command("sudo -u root ls /tmp", config=cfg)
    assert is_safe is True
    assert "validated" in reason.lower()


@pytest.mark.asyncio
async def test_stt_is_speech_int16_overflow():
    """Verify STTEngine._is_speech does not overflow on large int16 audio samples."""
    cfg = Config()
    stt = STTEngine(cfg)
    
    # Large int16 samples that would overflow 16-bit integer squaring (> 181)
    import numpy as np
    large_samples = np.array([30000, -30000, 25000, -25000], dtype=np.int16)
    audio_data = large_samples.tobytes()
    
    # Should calculate energy without raising RuntimeWarning or NaN
    is_speech = stt._is_speech(audio_data)
    assert isinstance(is_speech, bool)


@pytest.mark.asyncio
async def test_vector_store_similarity_search_unattached_llm():
    """Verify VectorStore returns empty results when self.llm is None."""
    vs = VectorStore(persist_directory="/tmp/test_vs_unattached", llm_engine=None)
    results = await vs.similarity_search("test query")
    assert results == []


@pytest.mark.asyncio
async def test_local_engine_embed_batch_asyncio_defined():
    """Verify LocalEngine.embed_batch resolves asyncio without NameError."""
    from friday.llm.local import LocalEngine
    
    engine = LocalEngine(primary_model="llama3.1:8b", fallback_model="phi3:mini")
    engine._known_embed_model = "nomic-embed-text"
    engine._client = AsyncMock()
    engine._client.embeddings = AsyncMock(return_value={"embedding": [0.1, 0.2, 0.3]})
    
    results = await engine.embed_batch(["text 1", "text 2"])
    assert len(results) == 2
    assert results[0] == [0.1, 0.2, 0.3]
