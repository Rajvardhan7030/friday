"""Plugin management and manifest schema."""

import os
import sys
import json
import yaml
import shutil
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Awaitable
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class PluginManifest(BaseModel):
    """Schema for a Friday plugin manifest."""
    name: str = Field(description="Unique name of the plugin")
    version: str = Field(description="Version string of the plugin")
    description: str = Field(default="", description="Short description of the plugin")
    author: str = Field(default="Unknown", description="Author of the plugin")
    entry_point: str = Field(description="Python module path to load (e.g., 'my_plugin.main')")
    type: str = Field(default="skill", description="Type of plugin: 'skill', 'agent', or 'theme'")
    permissions: List[str] = Field(default_factory=list, description="List of required permissions (e.g., 'network:github.com')")
    config_schema: Dict[str, Any] = Field(default_factory=dict, description="JSON schema for plugin configuration")
    min_friday_version: Optional[str] = Field(None, description="Minimum Friday version required")
    enabled: bool = Field(default=True, description="Whether the plugin is enabled")

class PluginManager:
    """Discovers and loads Friday plugins."""

    def __init__(self, plugin_dirs: Optional[List[Path]] = None):
        self.plugins: Dict[str, PluginManifest] = {}
        self.plugin_dirs = plugin_dirs or []
        
        # Add default user plugin dir
        self.user_plugin_dir = Path.home() / ".friday" / "plugins"
        self.user_plugin_dir.mkdir(parents=True, exist_ok=True)
        if self.user_plugin_dir not in self.plugin_dirs:
            self.plugin_dirs.append(self.user_plugin_dir)
            
        # Add builtin plugin dir
        builtin_plugin_dir = Path(__file__).parent.parent / "plugins"
        if builtin_plugin_dir.exists() and builtin_plugin_dir not in self.plugin_dirs:
            self.plugin_dirs.append(builtin_plugin_dir)

    def discover_plugins(self):
        """Scans directories for plugin manifests and loads them."""
        for p_dir in self.plugin_dirs:
            if not p_dir.exists() or not p_dir.is_dir():
                continue
            
            # Add plugin dir to sys.path so modules can be imported
            if str(p_dir) not in sys.path:
                sys.path.insert(0, str(p_dir))
                
            for entry in p_dir.iterdir():
                if entry.is_dir():
                    manifest_path = self._find_manifest(entry)
                    if manifest_path:
                        self.load_plugin_manifest(manifest_path)

    def _find_manifest(self, plugin_dir: Path) -> Optional[Path]:
        for filename in ["manifest.json", "plugin.yaml", "plugin.yml"]:
            path = plugin_dir / filename
            if path.exists():
                return path
        return None

    def load_plugin_manifest(self, manifest_path: Path):
        """Load a single plugin manifest."""
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                if manifest_path.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            manifest = PluginManifest(**data)
            self.plugins[manifest.name] = manifest
            
            if not manifest.enabled:
                logger.info(f"Plugin '{manifest.name}' is disabled.")
                return

            # Dynamically import the entry point
            try:
                importlib.import_module(manifest.entry_point)
                logger.info(f"Successfully loaded plugin: {manifest.name} v{manifest.version}")
            except ImportError as e:
                logger.error(f"Failed to import entry point '{manifest.entry_point}' for plugin '{manifest.name}': {e}")
                
        except Exception as e:
            logger.error(f"Error loading plugin manifest from {manifest_path}: {e}")

    def install_plugin(self, source_path: Path) -> bool:
        """Copy a plugin directory to the user plugin folder."""
        try:
            if not source_path.is_dir():
                logger.error(f"Plugin source must be a directory: {source_path}")
                return False
            
            dest_path = self.user_plugin_dir / source_path.name
            if dest_path.exists():
                shutil.rmtree(dest_path)
                
            shutil.copytree(source_path, dest_path)
            logger.info(f"Installed plugin to {dest_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to install plugin: {e}")
            return False

# Global plugin manager instance
plugin_manager = PluginManager()
