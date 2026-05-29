"""
Agent Router for FRIDAY.
Orchestrates intent detection and routing to specialized agents.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Type, Union
from .base import BaseAgent, Context, AgentResult
from ..llm.engine import LLMEngine, Message
from ..llm.router import ModelRouter
from ..core.config import Config

logger = logging.getLogger(__name__)

class AgentRouter:
    """
    The central hub for FRIDAY's multi-agent system.
    Detects user intent and delegates to the appropriate specialized agent.
    """

    def __init__(self, llm_engine: Union[LLMEngine, ModelRouter], config: Optional[Config] = None):
        self.config = config or Config()
        self._agents: Dict[str, BaseAgent] = {}
        self._fallback_agent_name = "general_chat"
        
        # Support both raw LLMEngine and ModelRouter
        if isinstance(llm_engine, ModelRouter):
            self.model_router = llm_engine
            self.llm = None # Will be lazily initialized
        else:
            self.llm = llm_engine
            self.model_router = None

    async def _get_llm(self, task_type: str = "general_chat") -> LLMEngine:
        """Lazily initialize or retrieve the LLM engine."""
        if self.llm:
            return self.llm
        if self.model_router:
            return await self.model_router.get_engine_for_task(task_type)
        raise ValueError("No LLM engine or ModelRouter available")

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
        target_agent_name = await self.detect_intent(user_query, chat_history)
        
        # 2. Delegate to Agent
        return await self.route_to(target_agent_name, user_query, chat_history)

    async def route_to(self, target_agent_name: str, user_query: str, chat_history: List[Dict[str, str]] = None) -> AgentResult:
        """
        Route to a specific agent by name.
        """
        chat_history = chat_history or []
        agent = self._agents.get(target_agent_name)
        
        if not agent:
            logger.info(f"Routing to fallback (general chat) for query: {user_query}")
            return await self._run_general_chat(user_query, chat_history)
        
        logger.info(f"Routing to agent '{target_agent_name}' for query: {user_query}")
        ctx = Context(user_query=user_query, chat_history=chat_history)
        return await agent.run(ctx)

    async def detect_intent(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Use the LLM to classify the user's intent based on agent descriptions.
        """
        # 1. Fast path: Detect greetings or identity questions without LLM
        fast_path = self._detect_fast_path_intent(query)
        if fast_path:
            return fast_path

        # 2. LLM-based classification
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
            llm = await self._get_llm("general_chat")
            # Limit response length for classification (num_predict is Ollama-specific, max_tokens for OpenAI)
            options = {"num_predict": 20, "max_tokens": 20, "temperature": 0.0}
            response = await llm.chat(messages, options=options)
            data = self._parse_json(response.content)
            return data.get("agent", "general_chat")
        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            
        return "general_chat"

    def _detect_fast_path_intent(self, query: str) -> Optional[str]:
        """Detect intent using simple regex patterns to avoid LLM calls."""
        import re
        greetings_regex = r"^(?:hi+|hello+|hey+|greetings|good (?:morning|evening)|morning|evening)(?:\s+friday)?[!.?]*$"
        identity_regex = r"^(?:who (?:are|r) (?:you|u)|what(?:'s| is) your name|your name)[?.!]*$"
        
        if re.search(greetings_regex, query, re.IGNORECASE) or re.search(identity_regex, query, re.IGNORECASE):
            return "general_chat"
        return None

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
        
        llm = await self._get_llm("general_chat")
        response = await llm.chat(messages)
        return AgentResult(content=response.content)
