"""Plugin management and manifest schema."""

import os
import sys
import json
import yaml
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

class PluginManager:
    """Discovers and loads Friday plugins."""

    def __init__(self, plugin_dirs: Optional[List[Path]] = None):
        self.plugins: Dict[str, PluginManifest] = {}
        self.plugin_dirs = plugin_dirs or []
        
        # Add default user plugin dir
        user_plugin_dir = Path.home() / ".friday" / "plugins"
        if user_plugin_dir.exists() and user_plugin_dir not in self.plugin_dirs:
            self.plugin_dirs.append(user_plugin_dir)
            
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
            
            # Dynamically import the entry point
            try:
                importlib.import_module(manifest.entry_point)
                logger.info(f"Successfully loaded plugin: {manifest.name} v{manifest.version}")
            except ImportError as e:
                logger.error(f"Failed to import entry point '{manifest.entry_point}' for plugin '{manifest.name}': {e}")
                
        except Exception as e:
            logger.error(f"Error loading plugin manifest from {manifest_path}: {e}")

# Global plugin manager instance
plugin_manager = PluginManager()
