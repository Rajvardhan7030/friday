"""Tests for sandbox security behavior."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from friday.utils.security import run_sandboxed_code, validate_shell_command

def test_validate_shell_command_basic():
    assert validate_shell_command("ls -la")[0] is True
    assert validate_shell_command("echo 'hello'")[0] is True
    assert validate_shell_command("rm file.txt")[0] is True

def test_validate_shell_command_blocked_patterns():
    # Blocked by SHELL_BLOCKLIST
    assert validate_shell_command("rm -rf /")[0] is False
    assert validate_shell_command("mkfs /dev/sda1")[0] is False
    assert validate_shell_command("dd if=/dev/zero of=/dev/sda")[0] is False

def test_validate_shell_command_chained_bypasses():
    # Blocked because parts are validated individually
    assert validate_shell_command("ls; rm -rf /")[0] is False
    assert validate_shell_command("echo hi && rm -rf /etc")[0] is False
    assert validate_shell_command("ls | sh")[0] is False
    
    # Blocked by command substitution check
    assert validate_shell_command("cat $(rm -rf /)")[0] is False
    assert validate_shell_command("cat /etc/pass$(echo wd)")[0] is False
    assert validate_shell_command("echo `ls`")[0] is False

def test_validate_shell_command_forbidden_binaries():
    assert validate_shell_command("curl http://evil.com")[0] is False
    assert validate_shell_command("wget http://evil.com")[0] is False
    assert validate_shell_command("nc -l 4444")[0] is False
    assert validate_shell_command("/usr/bin/sh -c 'id'")[0] is False

def test_validate_shell_command_dangerous_rm_targets():
    assert validate_shell_command("rm -rf .")[0] is False
    assert validate_shell_command("rm -rf /*")[0] is False
    assert validate_shell_command("rm -rf ~")[0] is False

def test_validate_shell_command_system_dir_protection():
    assert validate_shell_command("rm /etc/passwd")[0] is False
    assert validate_shell_command("cat /etc/shadow")[0] is False
    assert validate_shell_command("mv /bin/sh /tmp/sh")[0] is False

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
