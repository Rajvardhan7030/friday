"""Tests for sandbox security behavior."""

import subprocess
from unittest.mock import MagicMock

from friday.utils.security import run_sandboxed_code


def test_unshare_backend_fails_closed_when_isolation_is_unavailable(monkeypatch, tmp_path):
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: {
        "security.sandbox_timeout": 30,
        "security.sandbox_backend": "unshare",
    }.get(key, default)

    popen = MagicMock()

    monkeypatch.setattr("friday.utils.security.shutil.which", lambda name: "/usr/bin/unshare")
    monkeypatch.setattr(
        "friday.utils.security.subprocess.run",
        MagicMock(side_effect=subprocess.CalledProcessError(1, ["unshare", "-n", "true"])),
    )
    monkeypatch.setattr("friday.utils.security.subprocess.Popen", popen)

    success, message = run_sandboxed_code("print('hi')", tmp_path, config=config)

    assert success is False
    assert "not permitted" in message
    popen.assert_not_called()


def test_none_backend_allows_local_execution(monkeypatch, tmp_path):
    config = MagicMock()
    config.get.side_effect = lambda key, default=None: {
        "security.sandbox_timeout": 30,
        "security.sandbox_backend": "none",
    }.get(key, default)

    process = MagicMock()
    process.communicate.return_value = ("ok", "")
    process.returncode = 0

    monkeypatch.setattr("friday.utils.security.subprocess.Popen", MagicMock(return_value=process))

    success, message = run_sandboxed_code("print('hi')", tmp_path, config=config)

    assert success is True
    assert message == "ok"
