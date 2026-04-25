import json
import logging
import os
import importlib.util
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class SkillManager:
    def __init__(self, skills_dir: str = "friday/skills"):
        self.skills_dir = skills_dir
        self.index_path = os.path.join(self.skills_dir, "skills_index.json")
        self._ensure_skills_dir()
        self.registry: Dict[str, Dict[str, Any]] = self._load_registry()

    def _ensure_skills_dir(self):
        os.makedirs(self.skills_dir, exist_ok=True)
        init_file = os.path.join(self.skills_dir, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("")

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading skill index: {e}")
        return {}

    def _save_registry(self):
        with open(self.index_path, "w") as f:
            json.dump(self.registry, f, indent=4)

    def register_skill(self, name: str, description: str, parameters: Dict[str, Any], filepath: str):
        self.registry[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "filepath": filepath
        }
        self._save_registry()
        logger.info(f"Registered new skill: {name}")

    def create_skill(self, name: str, code: str, description: str, parameters: Dict[str, Any]):
        filepath = os.path.join(self.skills_dir, f"{name}.py")
        with open(filepath, "w") as f:
            f.write(code)
        self.register_skill(name, description, parameters, filepath)

    async def execute_skill(self, name: str, **kwargs) -> str:
        if name not in self.registry:
            return f"Error: Skill '{name}' not found."

        skill_info = self.registry[name]
        filepath = skill_info["filepath"]
        
        if not os.path.exists(filepath):
            return f"Error: Skill file '{filepath}' missing."

        try:
            spec = importlib.util.spec_from_file_location(name, filepath)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "run"):
                    import asyncio
                    result = module.run(**kwargs)
                    if asyncio.iscoroutine(result):
                        return await result
                    return str(result)
                else:
                    return f"Error: Skill '{name}' does not implement 'run(**kwargs)'."
        except Exception as e:
            error_msg = f"Skill execution failed for '{name}': {e}"
            logger.error(error_msg)
            # In a full system, you would trigger the auto-repair logic here
            return error_msg
        
        return "Unknown error executing skill."

    def get_skill_list(self) -> str:
        if not self.registry:
            return "No skills available."
        skills = []
        for name, info in self.registry.items():
            skills.append(f"- {name}: {info['description']} (params: {info['parameters']})")
        return "\n".join(skills)
