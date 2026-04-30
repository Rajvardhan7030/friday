"""Orchestrator for managing sessions and executing commands."""

import logging
import pkgutil
import asyncio
from importlib import import_module
from typing import Optional, List, Dict, Any
from pathlib import Path

from .registry import registry
from .plugin import plugin_manager
from .config import Config
from .exceptions import ModelNotFoundError
from ..llm.local import LocalEngine
from ..llm.api import APIEngine
from ..llm.engine import Message
from ..memory.document_indexer import DocumentIndexer
from ..memory.vector_store import VectorStore
from ..voice.tts import TTSEngine

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
        self._summarize_lock = asyncio.Lock()

    def add_message(self, role: str, content: str, llm: Optional['LocalEngine'] = None):
        """Add a message to the session history."""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history_messages:
            overflow = len(self.history) - self.max_history_messages
            archived_messages = self.history[:overflow]
            self.history = self.history[overflow:]
            if llm:
                # Trigger semantic summarization in background
                asyncio.create_task(self._summarize_messages(archived_messages, llm))
            else:
                self._append_to_summary(archived_messages)

    async def _summarize_messages(self, archived_messages: List[Dict[str, str]], llm: 'LocalEngine') -> None:
        """Condense evicted messages into a structured semantic JSON summary."""
        import json

        async with self._summarize_lock:
            lines = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in archived_messages]
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

    def build_llm_messages(self, user_input: str) -> List[Message]:
        """Build the message list for free-form chat with summarized context."""
        messages: List[Message] = []
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
        messages.extend(Message(role=entry["role"], content=entry["content"]) for entry in recent_history)
        messages.append(Message(role="user", content=user_input))
        return messages

    def _append_to_summary(self, archived_messages: List[Dict[str, str]]) -> None:
        """Condense evicted messages into a rolling plain-text summary."""
        summary_lines = [self.history_summary] if self.history_summary else []
        summary_lines.extend(
            f"{message['role'].capitalize()}: {message['content']}"
            for message in archived_messages
        )
        combined_summary = "\n".join(line for line in summary_lines if line)
        self.history_summary = combined_summary[-self.summary_max_chars:]

class AgentRunner:
    """The 'Brain' that decides how to handle an input."""

    def __init__(self, config: Config):
        self.config = config
        self.session = Session(
            max_history_messages=config.get("session.max_history_messages", 100),
            recent_messages=config.get("session.recent_messages", 20),
            summary_max_chars=config.get("session.summary_max_chars", 4000),
        )
        self.vector_store: Optional[VectorStore] = None
        self.document_indexer: Optional[DocumentIndexer] = None
        self._memory_ready = False
        self._memory_disabled_reason: Optional[str] = None

        try:
            engine_type = config.get("llm.engine", "ollama")
            if engine_type == "openai":
                self.llm = APIEngine(
                    model_name=config.get("llm.primary_model"),
                    api_key=config.get("llm.api_key"),
                    base_url=config.get("llm.api_base_url", "https://api.openai.com/v1")
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
        self._setup_memory()

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

    def _setup_memory(self) -> None:
        """Prepare long-term memory components for lazy initialization."""
        if not self.config.get("memory.enabled", True):
            self._memory_disabled_reason = "Memory is disabled by configuration."
            return

        if self.llm is None:
            self._memory_disabled_reason = "LLM engine unavailable for memory embeddings."
            return

        persist_directory = self.config.get("memory.persist_directory")
        self.vector_store = VectorStore(persist_directory, self.llm)
        self.document_indexer = DocumentIndexer(self.vector_store)

    async def _ensure_memory_ready(self) -> None:
        """Initialize the vector store and auto-index configured directories once."""
        if self._memory_ready or self.vector_store is None or self.document_indexer is None:
            return

        try:
            await self.vector_store.initialize()
            await self._auto_index_memory_directories()
            self._memory_ready = True
        except Exception as e:
            self._memory_disabled_reason = str(e)
            self.vector_store = None
            self.document_indexer = None
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

        memory_results = await self.vector_store.similarity_search(
            text,
            k=self.config.get("memory.retrieval_limit", 3),
        )
        if not memory_results:
            return None

        memory_lines = []
        for result in memory_results:
            source = result["metadata"].get("source", "memory")
            memory_lines.append(f"Source: {source}\nContent: {result['content']}")

        return Message(
            role="system",
            content="Relevant long-term memory:\n\n" + "\n\n".join(memory_lines),
        )

    async def _remember_exchange(self, user_text: str, assistant_text: str) -> None:
        """Persist completed exchanges for future retrieval."""
        if not self.config.get("memory.auto_remember_conversations", True):
            return

        await self._ensure_memory_ready()
        if self.vector_store is None:
            return

        document = f"User: {user_text}\nAssistant: {assistant_text}"
        metadata = {"source": "conversation", "type": "chat_exchange"}
        doc_id = f"chat_{len(self.session.history)}"
        await self.vector_store.add_documents([document], [metadata], [doc_id])

    async def handle_input(self, text: str) -> str:
        """Main entry point for processing any user input."""
        text = text.strip()
        if not text:
            return "I'm listening, but I didn't hear anything."

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
                self.session.add_message("user", text, llm=self.llm)
                self.session.add_message("assistant", str(result))
                await self._remember_exchange(text, str(result))
                return result
            except Exception as e:
                logger.error(f"Command {cmd.name} failed: {e}", exc_info=True)
                return f"I encountered an error running '{cmd.name}': {str(e)}"

        # 2. Fallback to LLM if no command matches
        return await self._fallback_to_llm(text)

    async def _fallback_to_llm(self, text: str) -> str:
        """Use the local LLM when no specific command is triggered, with MCP tool support."""
        from .mcp import mcp_client
        import json
        
        logger.info("No command match. Falling back to LLM.")
        if self.llm is None:
            return "The local LLM engine is unavailable. Install the required dependencies and start Ollama to enable free-form chat."
        
        # Build chat history for LLM
        messages = self.session.build_llm_messages(text)
        memory_message = await self._build_memory_message(text)
        if memory_message is not None:
            insert_at = 1 if messages and messages[0].role == "system" else 0
            messages.insert(insert_at, memory_message)
        
        self.session.add_message("user", text, llm=self.llm)
        
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
                    messages.append(Message(role="assistant", content=content))
                    for tool_call in tool_calls:
                        func_name = tool_call.get("function", {}).get("name")
                        func_args_str = tool_call.get("function", {}).get("arguments", "{}")
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
                                
                            messages.append(Message(role="tool", content=tool_result_str))
                        except Exception as e:
                            logger.error(f"Tool execution failed: {e}")
                            messages.append(Message(role="tool", content=f"Error executing tool: {str(e)}"))
                    continue # Loop back to LLM with tool responses
                else:
                    # No more tool calls, we have our final answer
                    self.session.add_message("assistant", content)
                    await self._remember_exchange(text, content)
                    return content
            
            # If max iterations reached
            final_msg = "I've reached my thinking limit on this task."
            self.session.add_message("assistant", final_msg, llm=self.llm)
            return final_msg
            
        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
            return f"I'm sorry, my brain is feeling a bit foggy: {str(e)}"
