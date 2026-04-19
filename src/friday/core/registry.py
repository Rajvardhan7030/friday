"""Dynamic Skill Registry for FRIDAY."""

import importlib.util
import inspect
import logging
import pkgutil
import sys
from pathlib import Path
from typing import Dict, Any, List, Type, Optional

from friday.skills.base import BaseSkill
from friday.core.exceptions import SkillError

logger = logging.getLogger(__name__)

class SkillRegistry:
    """Registry that handles skill discovery and loading."""

    def __init__(self, user_skills_dir: Optional[Path] = None):
        self.skills: Dict[str, BaseSkill] = {}
        self.user_skills_dir = user_skills_dir

    def register(self, skill: BaseSkill) -> None:
        """Manually register a skill instance."""
        if skill.name in self.skills:
            logger.warning(f"Overwriting skill: {skill.name}")
        self.skills[skill.name] = skill

    def discover_built_in(self) -> None:
        """Discover skills from friday.skills package."""
        import friday.skills as skills_pkg
        
        # Iterating over submodules in the skills package
        for _, name, is_pkg in pkgutil.iter_modules(skills_pkg.__path__):
            if is_pkg or name == "base":
                continue
                
            module_name = f"friday.skills.{name}"
            try:
                module = importlib.import_module(module_name)
                self._extract_skills_from_module(module)
            except Exception as e:
                logger.error(f"Failed to load built-in skill module {module_name}: {e}")

    def discover_user_skills(self) -> None:
        """Discover skills from the user's ~/.friday/skills directory."""
        if not self.user_skills_dir or not self.user_skills_dir.exists():
            return

        for item in self.user_skills_dir.iterdir():
            if item.is_file() and item.suffix == ".py" and item.name != "__init__.py":
                # SECURE: Load without polluting sys.path or allowing shadowing
                module_name = f"friday.user_skills.{item.stem}"
                try:
                    spec = importlib.util.spec_from_file_location(module_name, item)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        # We don't need to add to sys.modules if we don't plan to reload/pickle
                        spec.loader.exec_module(module)
                        self._extract_skills_from_module(module)
                except Exception as e:
                    logger.error(f"Failed to load user skill {item.name}: {e}")

    def _extract_skills_from_module(self, module: Any) -> None:
        """Extract classes inheriting from BaseSkill from a module."""
        for _, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj) and 
                issubclass(obj, BaseSkill) and 
                obj is not BaseSkill
            ):
                try:
                    skill_instance = obj()
                    self.register(skill_instance)
                    logger.debug(f"Successfully registered skill: {skill_instance.name}")
                except Exception as e:
                    logger.error(f"Failed to instantiate skill class {obj.__name__}: {e}")

    def get_skill(self, name: str) -> BaseSkill:
        """Retrieve a skill by name."""
        if name not in self.skills:
            raise SkillError(f"Skill '{name}' not found.")
        return self.skills[name]

    def list_skills(self) -> List[Dict[str, str]]:
        """List all registered skills."""
        return [
            {"name": s.name, "description": s.description}
            for s in self.skills.values()
        ]
