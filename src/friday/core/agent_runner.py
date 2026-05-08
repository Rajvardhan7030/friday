"""Orchestrator for managing sessions and executing commands."""

import logging
import pkgutil
import asyncio
import json
import uuid
from importlib import import_module
from typing import Optional, List, Dict, Any, Set
from pathlib import Path

from .registry import registry
from .plugin import plugin_manager
from .config import Config
from .exceptions import ModelNotFoundError
from ..llm.api import create_api_engine
from ..llm.engine import Message
from ..llm.local import LocalEngine
from ..memory.consolidator import MemoryConsolidator
from ..memory.conversation import ConversationMemory
from ..memory.document_indexer import DocumentIndexer
from ..memory.vector_store import VectorStore
from ..voice.tts import TTSEngine
from ..agents.adaptive_rag import AdaptiveRAGAgent
from ..agents.router import AgentRouter
from ..agents.tools import LocalDocumentRetriever

logger = logging.getLogger(__name__)

class Session:
    """Conversation state with rolling summary support."""

    def __init__(
        self,
        max_history_messages: int = 100,
        recent_messages: int = 20,
        summary_max_chars: int = 4000,
    ):
        self.history: List[Dict[str, str]] = []
        self.last_action: Optional[str] = None
        self.variables: Dict[str, Any] = {}
        self.max_history_messages = max_history_messages
        self.recent_messages = recent_messages
        self.summary_max_chars = summary_max_chars
        self.history_summary = ""
        self.session_id = uuid.uuid4().hex[:8]
        self._summarize_lock = asyncio.Lock()
        self._pending_tasks: Set[asyncio.Task] = set()

    async def aclose(self) -> None:
        """Wait for all pending summarization tasks to complete."""
        if self._pending_tasks:
            logger.info(f"Waiting for {len(self._pending_tasks)} pending summarization tasks...")
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()

    def add_message(self, role: str, content: Optional[str] = None, llm: Optional['LocalEngine'] = None, **kwargs):
        """Add a message to the session history."""
        msg = {"role": role, "content": content}
        msg.update(kwargs)
        self.history.append(msg)
        if len(self.history) > self.max_history_messages:
            overflow = len(self.history) - self.max_history_messages
            archived_messages = self.history[:overflow]
            self.history = self.history[overflow:]
            if llm:
                # Trigger semantic summarization in background and track it
                task = asyncio.create_task(self._summarize_messages(archived_messages, llm))
                self._pending_tasks.add(task)
                task.add_done_callback(self._pending_tasks.discard)
            else:
                self._append_to_summary(archived_messages)

    async def _summarize_messages(self, archived_messages: List[Dict[str, Any]], llm: 'LocalEngine') -> None:
        """Condense evicted messages into a structured semantic JSON summary."""
        async with self._summarize_lock:
            lines = []
            for msg in archived_messages:
                role = msg['role'].capitalize()
                content = msg.get('content') or ""
                if msg.get('tool_calls'):
                    content += f" [Calls tools: {', '.join(tc.get('function', {}).get('name', '') for tc in msg['tool_calls'])}]"
                lines.append(f"{role}: {content}")
            
            chat_text = "\n".join(lines)

            current = self.history_summary if self.history_summary else "{}"
            prompt = (
                "You are an AI assistant's memory manager. Analyze this conversation snippet "
                "and update the current summary. The summary MUST be valid JSON containing "
                "'user_preferences' (list), 'active_tasks' (list), and 'general_context' (string).\n\n"
                f"Current Summary:\n{current}\n\n"
                f"New Messages:\n{chat_text}\n\n"
                "Return ONLY the updated JSON object. Do not include markdown blocks or extra text."
            )

            try:
                res = await llm.chat([Message(role="system", content=prompt)])
                content = res.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                elif content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                # Verify JSON
                json.loads(content)
                self.history_summary = content
                logger.info("Session semantic memory updated successfully.")
            except Exception as e:
                logger.warning(f"Semantic summarization failed, falling back to text: {e}")
                self._append_to_summary(archived_messages)

    def build_llm_messages(self) -> List[Message]:
        """Build the message list for free-form chat with summarized context."""
        messages: List[Message] = [
            Message(
                role="system",
                content="You are FRIDAY, a helpful, privacy-first local AI assistant. Answer the user's request directly or use tools if needed."
            )
        ]
        if self.history_summary:
            messages.append(
                Message(
                    role="system",
                    content=(
                        "Conversation summary from earlier in this session:\n"
                        f"{self.history_summary}"
                    ),
                )
            )

        recent_history = self.history[-self.recent_messages:] if self.recent_messages > 0 else []
        
        valid_fields = {"role", "content", "name", "tool_calls", "tool_call_id"}
        for entry in recent_history:
            # Filter only valid fields to avoid Pydantic ValidationError with extra data
            filtered_entry = {k: v for k, v in entry.items() if k in valid_fields}
            messages.append(Message(**filtered_entry))
        return messages

    def _append_to_summary(self, archived_messages: List[Dict[str, Any]]) -> None:
        """Condense evicted messages into a rolling plain-text summary."""
        summary_lines = [self.history_summary] if self.history_summary else []
        for message in archived_messages:
            role = message['role'].capitalize()
            content = message.get('content') or ""
            if message.get('tool_calls'):
                content += f" [Calls tools: {', '.join(tc.get('function', {}).get('name', '') for tc in message['tool_calls'])}]"
            summary_lines.append(f"{role}: {content}")
            
        combined_summary = "\n".join(line for line in summary_lines if line)
        self.history_summary = combined_summary[-self.summary_max_chars:]

class AgentRunner:
    """The 'Brain' that decides how to handle an input."""
    
    _mcp_ready: bool = False
    _mcp_lock: asyncio.Lock = None

    def __init__(self, config: Config):
        self.config = config
        self.session = Session(
            max_history_messages=config.get("session.max_history_messages", 100),
            recent_messages=config.get("session.recent_messages", 20),
            summary_max_chars=config.get("session.summary_max_chars", 4000),
        )
        self.vector_store: Optional[VectorStore] = None
        self.conversation_memory: Optional[ConversationMemory] = None
        self.memory_consolidator: Optional[MemoryConsolidator] = None
        self.document_indexer: Optional[DocumentIndexer] = None
        self._memory_ready = False
        self._memory_disabled_reason: Optional[str] = None
        self._memory_lock = asyncio.Lock()
        self._mcp_ready = False
        self._mcp_lock = asyncio.Lock()

        try:
            engine_type = config.get("llm.engine", "ollama")
            api_key = config.get("llm.api_key")
            provider = config.get("llm.provider", "ollama")
            
            # Use API engine if explicitly requested, or if an API key is present with a non-Ollama provider
            if engine_type == "openai" or (api_key and provider != "ollama"):
                self.llm = create_api_engine(
                    model_name=config.get("llm.primary_model"),
                    api_key=api_key,
                    base_url=config.get("llm.api_base_url", "https://api.openai.com/v1"),
                    embedding_model_name=config.get("llm.embedding_model")
                )
            else:
                self.llm = LocalEngine(
                    primary_model=config.get("llm.primary_model"),
                    fallback_model=config.get("llm.fallback_model"),
                    base_url=config.get("llm.base_url")
                )
        except Exception as e:
            logger.warning("LLM engine unavailable during startup: %s", e)
            self.llm = None

        try:
            self.tts = TTSEngine(config)
        except Exception as e:
            logger.warning("TTS engine unavailable during startup: %s", e)
            self.tts = None
        
        self._load_agents()
        self._load_skills()
        self._setup_memory()
        self.router: Optional[AgentRouter] = None
        self._setup_router()

    def _load_skills(self):
        """Initialize and register skills as MCP tools."""
        from ..skills.web_search_skill import WebSearchSkill
        from ..skills.browser_skill import BrowserSkill
        
        # Initialize skills
        self.skills = {
            "web_search": WebSearchSkill(),
            "browser_control": BrowserSkill(self.config.get("skills.browser.daemon_url", "http://localhost:9000"))
        }
        
        # Register as MCP tools
        for skill in self.skills.values():
            try:
                skill.register_mcp()
                logger.info(f"Skill '{skill.name}' registered as MCP tool.")
            except Exception as e:
                logger.warning(f"Failed to register skill '{skill.name}': {e}")

    def _setup_router(self):

        """Initialize the agent router and register specialized agents."""
        if self.llm is None:
            return

        self.router = AgentRouter(self.llm, self.config)

        # 1. Register Adaptive RAG
        if self.vector_store and self.document_indexer:
            retriever = LocalDocumentRetriever(self.vector_store, self.document_indexer)
            self.router.register_agent(AdaptiveRAGAgent(self.llm, retriever))

        # 2. Register Code Assistant
        from ..agents.code_assistant import CodeAssistantAgent
        from ..agents.sandbox_executor import SandboxExecutor
        self.router.register_agent(
            CodeAssistantAgent(self.llm, SandboxExecutor(self.config))
        )

        # 3. Register System Command Agent
        from ..agents.system_command_agent import SystemCommandAgent
        self.router.register_agent(SystemCommandAgent(self.llm, self.config))

        # 4. Register Research Agent from plugins
        try:
            from ..plugins.research.main import ResearchAgent
            if self.vector_store:
                self.router.register_agent(ResearchAgent(self.llm, self.vector_store))
        except (ImportError, ModuleNotFoundError):
            logger.debug("ResearchAgent plugin not loaded into router.")

    def _load_agents(self):

        """Discover and import agent modules to trigger registry decorators."""
        try:
            agents_package = import_module("friday.agents")
        except ImportError as e:
            logger.warning("Could not import agents package: %s", e)
            agents_package = None

        if agents_package:
            skipped_modules = set(self.config.get("agents.skip_modules", []))

            for module_info in pkgutil.iter_modules(agents_package.__path__):
                module_name = module_info.name

                if module_name.startswith("_") or module_name in skipped_modules:
                    continue

                try:
                    import_module(f"friday.agents.{module_name}")
                except ImportError as e:
                    logger.warning("Could not load agent module %s: %s", module_name, e)

        # Discover and load dynamic plugins
        plugin_manager.discover_plugins()

    async def aclose(self):
        """Shutdown engines and close connections."""
        # Ensure all pending session tasks (like summarization) complete
        await self.session.aclose()

        if self.memory_consolidator and self.session.session_id:
            logger.info(f"Consolidating memory for session {self.session.session_id} before shutdown...")
            try:
                await self.memory_consolidator.consolidate_session(self.session.session_id)
            except Exception as e:
                logger.error(f"Memory consolidation failed during shutdown: {e}")

        if self.llm:
            await self.llm.aclose()
        if self.tts:
            await self.tts.aclose()
        
        from .mcp import mcp_client
        await mcp_client.shutdown()
        
        logger.info("AgentRunner resources closed.")

    async def _ensure_mcp_ready(self) -> None:
        """Initialize external MCP servers once."""
        if self._mcp_lock is None:
            self._mcp_lock = asyncio.Lock()
            
        async with self._mcp_lock:
            if self._mcp_ready:
                return
            
            from .mcp import mcp_client
            server_configs = self.config.get("mcp_servers", {})
            if server_configs:
                logger.info(f"Initializing {len(server_configs)} external MCP servers...")
                await mcp_client.initialize_external_servers(server_configs)
            
            self._mcp_ready = True

    def _setup_memory(self) -> None:
        """Prepare long-term memory components for lazy initialization."""
        if not self.config.get("memory.enabled", True):
            self._memory_disabled_reason = "Memory is disabled by configuration."
            return

        if self.llm is None:
            self._memory_disabled_reason = "LLM engine unavailable for memory embeddings."
            return

        persist_directory = self.config.get("memory.persist_directory")
        db_path = Path(persist_directory) / "conversation.db"
        ltm_collection = self.config.get("memory.ltm_collection", "ltm_memory")
        
        self.vector_store = VectorStore(persist_directory, self.llm)
        self.conversation_memory = ConversationMemory(str(db_path))
        self.memory_consolidator = MemoryConsolidator(
            self.llm, 
            self.vector_store, 
            self.conversation_memory,
            ltm_collection=ltm_collection
        )
        self.document_indexer = DocumentIndexer(self.vector_store)

    async def _ensure_memory_ready(self) -> None:
        """Initialize the vector store and auto-index configured directories once."""
        if self._memory_ready or self.vector_store is None or self.document_indexer is None:
            return

        async with self._memory_lock:
            # Re-check inside the lock
            if self._memory_ready:
                return
                
            try:
                await self.vector_store.initialize()
                if self.conversation_memory:
                    await self.conversation_memory.initialize()
                
                # Background indexing: don't wait for it to complete before continuing
                # This prevents 'friday ask' from hanging while large directories are indexed.
                asyncio.create_task(self._auto_index_memory_directories())
                
                self._memory_ready = True
            except Exception as e:
                self._memory_disabled_reason = str(e)
                self.vector_store = None
                self.document_indexer = None
                self.conversation_memory = None
                self.memory_consolidator = None
                logger.warning("Long-term memory unavailable: %s", e)

    async def _auto_index_memory_directories(self) -> None:
        """Index configured directories for retrieval-augmented responses."""
        if self.document_indexer is None:
            return

        for raw_path in self.config.get("memory.auto_index_directories", []):
            path = Path(raw_path).expanduser()
            if not path.exists():
                continue
            await self.document_indexer.index_directory(path)

    async def _build_memory_message(self, text: str) -> Optional[Message]:
        """Retrieve relevant long-term memory for the current query."""
        await self._ensure_memory_ready()
        if self.vector_store is None:
            return None

        # Pre-compute embedding once to avoid redundant LLM calls
        query_embedding = await self.llm.embed(text)

        # Search MTM (Conversations)
        mtm_results = await self.vector_store.similarity_search(
            text,
            k=self.config.get("memory.retrieval_limit", 3),
            query_embedding=query_embedding
        )
        
        # Search LTM (Extracted Facts)
        ltm_results = await self.vector_store.similarity_search(
            text,
            k=2,
            collection_name=self.config.get("memory.ltm_collection", "ltm_memory"),
            query_embedding=query_embedding
        )
        
        if not mtm_results and not ltm_results:
            return None

        memory_lines = []
        for result in ltm_results:
            memory_lines.append(f"Factual Knowledge: {result['content']}")
        
        for result in mtm_results:
            source = result["metadata"].get("source", "memory")
            memory_lines.append(f"Recent History (Source: {source}): {result['content']}")

        return Message(
            role="system",
            content="Relevant long-term memory and factual knowledge:\n\n" + "\n\n".join(memory_lines),
        )

    async def _add_to_history(self, role: str, content: Optional[str] = None, **kwargs) -> None:
        """Add a message to both session history and persistent storage."""
        # 1. In-memory session history
        if hasattr(self, "session"):
            self.session.add_message(role, content, llm=getattr(self, "llm", None), **kwargs)
        
        # 2. Persistent SQLite history
        if hasattr(self, "config") and self.config.get("memory.enabled", True):
            await self._ensure_memory_ready()
            if getattr(self, "conversation_memory", None):
                # Metadata for the database (excludes fields already stored in separate columns)
                metadata = kwargs.copy()
                metadata.pop("llm", None)
                await self.conversation_memory.add_message(
                    self.session.session_id, 
                    role, 
                    content or "", 
                    metadata=metadata if metadata else None
                )

    async def _remember_exchange(self, user_text: str, assistant_text: str) -> None:
        """Persist completed exchanges for future retrieval (Vector Store only)."""
        if not self.config.get("memory.auto_remember_conversations", True):
            return

        await self._ensure_memory_ready()
        if self.vector_store is None:
            return

        # Store in MTM (Vector Store - Chat Exchange)
        document = f"User: {user_text}\nAssistant: {assistant_text}"
        metadata = {
            "source": "conversation",
            "type": "chat_exchange",
            "session_id": self.session.session_id
        }
        doc_id = f"chat_{self.session.session_id}_{len(self.session.history)}"
        await self.vector_store.add_documents([document], [metadata], [doc_id])

    async def handle_input(self, text: str) -> str:
        """Main entry point for processing any user input."""
        text = text.strip()
        self._last_tts_content = None # Reset for each interaction
        if not text:
            return "I'm listening, but I didn't hear anything."

        # Standardized: Add user message to history immediately
        await self._add_to_history("user", text)

        # 1. Check the Command Registry (Deterministic Logic)
        handler_data = registry.find_handler(text)
        if handler_data:
            cmd, match = handler_data
            logger.info(f"Executing command: {cmd.name}")
            try:
                # Pass session and regex matches to the handler
                # Some handlers might need self.llm or self.config
                result = await cmd.handler(
                    self.session,
                    *match.groups(),
                    llm=self.llm,
                    config=self.config,
                    tts=self.tts,
                )
                await self._add_to_history("assistant", str(result))
                await self._remember_exchange(text, str(result))
                return str(result)
            except Exception as e:
                logger.error(f"Command {cmd.name} failed: {e}", exc_info=True)
                return f"I encountered an error running '{cmd.name}': {str(e)}"

        # 2. Check for Specialized Agent Intent (AI Routing)
        if self.router:
            try:
                # Ask the router to classify the intent
                intent = await self.router._detect_intent(text, self.session.history)
                if intent in self.router._agents:
                    logger.info(f"Routing to specialized agent: {intent}")
                    agent_result = await self.router.route(text, self.session.history)
                    
                    self._last_tts_content = agent_result.metadata.get("tts_content")
                    await self._add_to_history("assistant", agent_result.content)
                    await self._remember_exchange(text, agent_result.content)
                    return agent_result.content
            except Exception as e:
                logger.error(f"Agent routing failed: {e}")
                # Fall through to general chat on router failure

        # 3. Fallback to LLM if no command matches
        return await self._fallback_to_llm(text)

    @property
    def last_tts_content(self) -> Optional[str]:
        """Returns the voice-friendly summary of the last agent response, if any."""
        return getattr(self, "_last_tts_content", None)

    async def _fallback_to_llm(self, text: str) -> str:
        """Use the local LLM when no specific command is triggered, with MCP tool support."""
        from .mcp import mcp_client
        import json
        
        # Ensure MCP servers are started
        await self._ensure_mcp_ready()
        
        logger.info("No command match. Falling back to LLM.")
        if self.llm is None or not self.llm.is_available():
            engine_type = self.config.get("llm.engine", "ollama")
            if engine_type == "openai":
                return "The API engine is unavailable. Please check your API key in config.yaml."
            return "The local LLM engine is unavailable. Install the required dependencies and start Ollama to enable free-form chat."
        
        # Build chat history for LLM
        messages = self.session.build_llm_messages()
        memory_message = await self._build_memory_message(text)
        if memory_message is not None:
            insert_at = 1 if messages and messages[0].role == "system" else 0
            messages.insert(insert_at, memory_message)
        
        tools = mcp_client.get_tools_for_llm()
        if not tools:
            tools = None
            
        try:
            # ReAct / Tool Loop
            max_iterations = 5
            for _ in range(max_iterations):
                response = await self.llm.chat(messages, tools=tools)
                content = response.content
                tool_calls = response.tool_calls
                
                if tool_calls:
                    # Assistant chose to call tools
                    assistant_msg = Message(role="assistant", content=content, tool_calls=tool_calls)
                    messages.append(assistant_msg)
                    await self._add_to_history(**assistant_msg.model_dump(exclude_none=True))
                    
                    for tool_call in tool_calls:
                        func_name = tool_call.get("function", {}).get("name")
                        func_args_str = tool_call.get("function", {}).get("arguments", "{}")
                        tool_call_id = tool_call.get("id")
                        try:
                            # Ollama sometimes returns string args, sometimes dict
                            if isinstance(func_args_str, str):
                                args = json.loads(func_args_str)
                            else:
                                args = func_args_str
                            
                            logger.info(f"LLM executing tool: {func_name} with args {args}")
                            tool_result = await mcp_client.call_tool(func_name, args)
                            
                            # Ensure tool result is a string
                            if isinstance(tool_result, dict) or isinstance(tool_result, list):
                                tool_result_str = json.dumps(tool_result)
                            else:
                                tool_result_str = str(tool_result)
                                
                            tool_msg = Message(
                                role="tool", 
                                content=tool_result_str, 
                                tool_call_id=tool_call_id,
                                name=func_name
                            )
                            messages.append(tool_msg)
                            await self._add_to_history(**tool_msg.model_dump(exclude_none=True))
                        except Exception as e:
                            logger.error(f"Tool execution failed: {e}")
                            error_msg = Message(
                                role="tool", 
                                content=f"Error executing tool: {str(e)}",
                                tool_call_id=tool_call_id,
                                name=func_name
                            )
                            messages.append(error_msg)
                            await self._add_to_history(**error_msg.model_dump(exclude_none=True))
                    continue # Loop back to LLM with tool responses
                else:
                    # No more tool calls, we have our final answer
                    await self._add_to_history("assistant", content)
                    await self._remember_exchange(text, content)
                    return content
            
            # If max iterations reached
            final_msg = "I've reached my thinking limit on this task."
            await self._add_to_history("assistant", final_msg)
            return final_msg
            
        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
            return f"I'm sorry, my brain is feeling a bit foggy: {str(e)}"
