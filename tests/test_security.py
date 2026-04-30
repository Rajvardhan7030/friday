"""Tests for sandbox security behavior."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from friday.utils.security import run_sandboxed_code


@pytest.mark.asyncio
async def test_unshare_backend_fails_closed_when_isolation_is_unavailable(monkeypatch, tmp_path):
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: {
        "security.sandbox_timeout": 30,
        "security.sandbox_backend": "unshare",
    }.get(key, default)

    monkeypatch.setattr("friday.utils.security.shutil.which", lambda name: "/usr/bin/unshare")
    
    # Mock asyncio.create_subprocess_exec for the validation check
    mock_proc_fail = AsyncMock()
    mock_proc_fail.returncode = 1
    mock_proc_fail.communicate.return_value = (b"", b"permission denied")
    
    mock_create_subprocess = AsyncMock(return_value=mock_proc_fail)
    monkeypatch.setattr("friday.utils.security.asyncio.create_subprocess_exec", mock_create_subprocess)

    success, message = await run_sandboxed_code("print('hi')", tmp_path, config=config)

    assert success is False
    assert "failed" in message or "not permitted" in message
    # Should have called create_subprocess_exec at least for the validation
    assert mock_create_subprocess.called


@pytest.mark.asyncio
async def test_none_backend_allows_local_execution(monkeypatch, tmp_path):
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: {
        "security.sandbox_timeout": 30,
        "security.sandbox_backend": "none",
    }.get(key, default)

    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (b"ok", b"")
    mock_proc.returncode = 0
    mock_proc.pid = 1234

    mock_create_subprocess = AsyncMock(return_value=mock_proc)
    monkeypatch.setattr("friday.utils.security.asyncio.create_subprocess_exec", mock_create_subprocess)

    success, message = await run_sandboxed_code("print('hi')", tmp_path, config=config)

    assert success is True
    assert message == "ok"
