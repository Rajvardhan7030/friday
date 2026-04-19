"""Dynamic Skill Registry for FRIDAY."""

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Dict, Any, List, Type, Optional

from src.friday.skills.base import BaseSkill
from src.friday.core.exceptions import SkillError

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
        """Discover skills from src.friday.skills package."""
        import src.friday.skills as skills_pkg
        
        # Iterating over submodules in the skills package
        for _, name, is_pkg in pkgutil.iter_modules(skills_pkg.__path__):
            if is_pkg or name == "base":
                continue
                
            module_name = f"src.friday.skills.{name}"
            try:
                module = importlib.import_module(module_name)
                self._extract_skills_from_module(module)
            except Exception as e:
                logger.error(f"Failed to load built-in skill module {module_name}: {e}")

    def discover_user_skills(self) -> None:
        """Discover skills from the user's ~/.friday/skills directory."""
        if not self.user_skills_dir or not self.user_skills_dir.exists():
            return

        # Simple dynamic loading for external skills
        import sys
        sys.path.append(str(self.user_skills_dir))

        for item in self.user_skills_dir.iterdir():
            if item.is_file() and item.suffix == ".py" and item.name != "__init__.py":
                module_name = item.stem
                try:
                    module = importlib.import_module(module_name)
                    self._extract_skills_from_module(module)
                except Exception as e:
                    logger.error(f"Failed to load user skill module {module_name}: {e}")

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
