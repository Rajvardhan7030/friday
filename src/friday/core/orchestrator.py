import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Set

from .config import Config
from .mcp import mcp_client
from ..skills.base import BaseSkill

logger = logging.getLogger(__name__)

class Orchestrator:
    """Handles external process lifecycle and tool registration."""

    def __init__(self, config: Config):
        self.config = config
        self._mcp_ready = False
        self._mcp_lock = asyncio.Lock()
        
        # Browser Daemon Management
        self._browser_daemon_process: Optional[subprocess.Popen] = None
        self._browser_daemon_lock = asyncio.Lock()

    async def ensure_mcp_ready(self, skills: Dict[str, BaseSkill]) -> None:
        """Initialize external MCP servers once."""
        await self._ensure_browser_daemon(skills)

        async with self._mcp_lock:
            if self._mcp_ready:
                return
            
            server_configs = self.config.get("mcp_servers", {})
            if server_configs:
                logger.info(f"Initializing {len(server_configs)} external MCP servers...")
                await mcp_client.initialize_external_servers(server_configs)
            
            self._mcp_ready = True

    async def _ensure_browser_daemon(self, skills: Dict[str, BaseSkill]) -> None:
        """Checks if the browser daemon is reachable, starts it if not."""
        skill = skills.get("browser_control")
        if not skill or not hasattr(skill, "is_daemon_alive"):
            return

        async with self._browser_daemon_lock:
            if await skill.is_daemon_alive():
                return

            logger.info("Browser daemon is offline. Attempting to start it...")
            
            # Find the binary
            project_root = Path(__file__).parent.parent.parent.parent
            daemon_dir = project_root / "src" / "friday" / "skills" / "browser_daemon"
            
            daemon_bin = None
            for bin_name in ["friday-browser-daemon", "daemon"]:
                path = daemon_dir / bin_name
                if path.exists():
                    daemon_bin = path
                    break
            
            if not daemon_bin:
                logger.error(f"Browser daemon binary not found in {daemon_dir}")
                return

            try:
                log_file = Path(self.config.get("logging.file")).parent / "browser_daemon.log"
                log_file.parent.mkdir(parents=True, exist_ok=True)
                out_file = open(log_file, "a")

                self._browser_daemon_process = subprocess.Popen(
                    [str(daemon_bin)],
                    cwd=str(daemon_dir),
                    stdout=out_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True
                )
                
                for _ in range(5):
                    await asyncio.sleep(1)
                    if await skill.is_daemon_alive():
                        logger.info("Browser daemon started successfully.")
                        return
                
                logger.warning("Browser daemon started but health check failed.")
            except Exception as e:
                logger.error(f"Failed to start browser daemon: {e}")

    async def shutdown(self):
        """Shutdown external servers and processes."""
        await mcp_client.shutdown()
        
        if self._browser_daemon_process:
            logger.info("Terminating browser daemon...")
            self._browser_daemon_process.terminate()
            try:
                self._browser_daemon_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._browser_daemon_process.kill()
            self._browser_daemon_process = None
