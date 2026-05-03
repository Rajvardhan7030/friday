"""
Agent Router for FRIDAY.
Orchestrates intent detection and routing to specialized agents.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Type
from .base import BaseAgent, Context, AgentResult
from ..llm.engine import LLMEngine, Message
from ..core.config import Config

logger = logging.getLogger(__name__)

class AgentRouter:
    """
    The central hub for FRIDAY's multi-agent system.
    Detects user intent and delegates to the appropriate specialized agent.
    """

    def __init__(self, llm_engine: LLMEngine, config: Optional[Config] = None):
        self.llm = llm_engine
        self.config = config or Config()
        self._agents: Dict[str, BaseAgent] = {}
        self._fallback_agent_name = "general_chat"

    def register_agent(self, agent: BaseAgent):
        """Register a specialized agent."""
        self._agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")

    async def route(self, user_query: str, chat_history: List[Dict[str, str]] = None) -> AgentResult:
        """
        Detect intent and route the query to the best agent.
        """
        chat_history = chat_history or []
        
        # 1. Detect Intent
        target_agent_name = await self._detect_intent(user_query, chat_history)
        
        # 2. Delegate to Agent
        agent = self._agents.get(target_agent_name)
        if not agent:
            logger.info(f"Routing to fallback (general chat) for query: {user_query}")
            return await self._run_general_chat(user_query, chat_history)
        
        logger.info(f"Routing to agent '{target_agent_name}' for query: {user_query}")
        ctx = Context(user_query=user_query, chat_history=chat_history)
        return await agent.run(ctx)

    async def _detect_intent(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Use the LLM to classify the user's intent based on agent descriptions.
        """
        agent_descriptions = "\n".join([
            f"- {name}: {agent.description}" 
            for name, agent in self._agents.items()
        ])
        
        system_prompt = f"""
You are the Intent Router for FRIDAY. Classify the user query into the most appropriate category.

Categories:
{agent_descriptions}
- general_chat: For greetings, general questions, or topics not covered above.

Guidelines:
- Use 'code_assistant' for: writing scripts, creating programs, executing Python code, "write a script to...", "create a python app that...".
- Use 'system_command' for: terminal tasks, file operations (ls, cd, mkdir), system info, "run command...", "list files in...".
- Use 'adaptive_rag' for: searching personal documents, "what does my notes say about...", "find info in my files".

Respond ONLY with JSON: {{"agent": "category_name"}}
"""
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"Query: {query}")
        ]
        
        try:
            response = await self.llm.chat(messages)
            data = self._parse_json(response.content)
            return data.get("agent", "general_chat")
        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            
        return "general_chat"

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Robustly extract JSON from potentially messy LLM output."""
        try:
            # Look for JSON block or raw curly braces
            match = re.search(r"(\{.*\})", text.replace("\n", " "), re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return json.loads(text)
        except Exception:
            return {}

    async def _run_general_chat(self, query: str, history: List[Dict[str, str]]) -> AgentResult:
        """Fallback for general conversation when no specialized agent matches."""
        messages = [
            Message(role="system", content="You are FRIDAY, a helpful, privacy-first local AI assistant.")
        ]
        # Add history (which now includes the current query)
        for msg in history[-5:]: # Last 5 turns for context
            messages.append(Message(role=msg["role"], content=msg["content"]))
        
        response = await self.llm.chat(messages)
        return AgentResult(content=response.content)

