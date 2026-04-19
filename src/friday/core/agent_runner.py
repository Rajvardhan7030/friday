"""Orchestrator for loading agents and managing execution context."""

import logging
from typing import Dict, Any, List, Optional, Type
from src.friday.agents.base import BaseAgent, Context, AgentResult
from src.friday.llm.engine import LLMEngine
from src.friday.core.registry import SkillRegistry
from src.friday.core.exceptions import AgentError

logger = logging.getLogger(__name__)

class AgentRunner:
    """Manages agents and routes queries to them."""

    def __init__(self, llm_engine: LLMEngine, skill_registry: SkillRegistry):
        self.llm = llm_engine
        self.registry = skill_registry
        self.agents: Dict[str, BaseAgent] = {}

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent instance."""
        self.agents[agent.name] = agent

    async def run_agent(
        self, 
        agent_name: str, 
        user_query: str, 
        history: Optional[List[Dict[str, str]]] = None
    ) -> AgentResult:
        """Run a specific agent by name."""
        if agent_name not in self.agents:
            raise AgentError(f"Agent '{agent_name}' not registered.")
        
        agent = self.agents[agent_name]
        ctx = Context(
            user_query=user_query,
            chat_history=history or [],
            metadata={"skills": self.registry.list_skills()}
        )
        
        try:
            return await agent.run(ctx)
        except Exception as e:
            logger.error(f"Error running agent {agent_name}: {e}")
            raise AgentError(f"Failed to execute agent: {e}")

    def list_agents(self) -> List[Dict[str, str]]:
        """List all registered agents."""
        return [
            {"name": a.name, "description": a.description}
            for a in self.agents.values()
        ]
