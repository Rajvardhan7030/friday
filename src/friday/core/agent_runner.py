"""Orchestrator for managing sessions and executing commands."""

import logging
import pkgutil
import asyncio
import json
from importlib import import_module
from typing import Optional, List, Dict, Any, Set, Union, AsyncIterator

from .registry import registry
from .plugin import plugin_manager
from .config import Config
from .exceptions import PermissionDeniedError, ProviderRateLimitError
from ..llm.engine import Message
from ..llm.router import ModelRouter
from ..agents.adaptive_rag import AdaptiveRAGAgent
from ..agents.code_assistant import CodeAssistantAgent
from ..agents.system_command_agent import SystemCommandAgent
from ..agents.sandbox_executor import SandboxExecutor
from ..agents.router import AgentRouter
from ..agents.tools import LocalDocumentRetriever
from ..skills.web_search_skill import WebSearchSkill
from ..skills.browser_skill import BrowserSkill
from .session import Session
from .mcp import mcp_client
from .observability import trace_manager
from .permissions import PermissionManager
from .recovery import recovery_manager
from .orchestrator import Orchestrator
from ..memory.manager import MemoryManager

logger = logging.getLogger(__name__)

class ToolExecutor:
    """Handles the execution of multiple tool calls from an LLM."""

    def __init__(self, mcp_client: Any, permission_manager: Optional[PermissionManager] = None):
        self.mcp_client = mcp_client
        self.permission_manager = permission_manager

    async def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Message]:
        """Execute a batch of tool calls and return a list of tool result messages."""
        results = []
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
                
                # Check permissions
                if self.permission_manager:
                    allowed = await self.permission_manager.check_permission(func_name, args)
                    if not allowed:
                        raise PermissionDeniedError(f"Permission denied for tool '{func_name}'")

                logger.info(f"LLM executing tool: {func_name} with args {args}")
                trace_manager.add_event("tool_called", {"tool": func_name, "args": args}, f"LLM calling tool: {func_name}")
                
                tool_result = await self.mcp_client.call_tool(func_name, args)
                
                # Ensure tool result is a string
                if isinstance(tool_result, (dict, list)):
                    tool_result_str = json.dumps(tool_result)
                else:
                    tool_result_str = str(tool_result)
                    
                trace_manager.add_event("tool_result", {"tool": func_name, "result": tool_result_str})
                
                results.append(Message(
                    role="tool", 
                    content=tool_result_str, 
                    tool_call_id=tool_call_id,
                    name=func_name
                ))
            except PermissionDeniedError as e:
                logger.warning(str(e))
                trace_manager.add_event("tool_denied", {"tool": func_name, "error": str(e)})
                results.append(Message(
                    role="tool", 
                    content=f"Error: {str(e)}",
                    tool_call_id=tool_call_id,
                    name=func_name
                ))
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                
                # Attempt Recovery
                recovered = await recovery_manager.attempt_recovery(e, {"tool_name": func_name})
                if recovered:
                    # Retry once if recovered (e.g. model pulled)
                    try:
                        tool_result = await self.mcp_client.call_tool(func_name, args)
                        tool_result_str = json.dumps(tool_result) if isinstance(tool_result, (dict, list)) else str(tool_result)
                        results.append(Message(role="tool", content=tool_result_str, tool_call_id=tool_call_id, name=func_name))
                        continue
                    except Exception:
                        pass

                error_msg = f"Error executing tool '{func_name}': {str(e)}"
                trace_manager.add_event("tool_failed", {"tool": func_name, "error": str(e)})
                # We still return a Message to the LLM so it can handle the error
                results.append(Message(
                    role="tool", 
                    content=error_msg,
                    tool_call_id=tool_call_id,
                    name=func_name
                ))
        return results

class AgentRunner:
    """The 'Brain' that decides how to handle an input."""
    
    def __init__(self, config: Config):
        self.config = config
        self.session = Session(
            max_history_messages=config.get("session.max_history_messages", 100),
            recent_messages=config.get("session.recent_messages", 20),
            summary_max_chars=config.get("session.summary_max_chars", 4000),
        )
        self._pending_tasks: Set[asyncio.Task] = set()
        
        # Specialized Services
        self.model_router = ModelRouter(config)
        self.memory_manager = MemoryManager(config, self.model_router)
        self.orchestrator = Orchestrator(config)
        self.permission_manager = PermissionManager(config)
        
        # Primary LLM for the runner (usually default/general_chat)
        self.llm: Optional[LLMEngine] = None
        
        self._load_agents()
        self._load_skills()
        self.router: Optional[AgentRouter] = None
        self._setup_router()

    async def _initialize_primary_llm(self):
        """Lazily initialize primary LLM to avoid blocking __init__."""
        if self.llm is not None:
            return
            
        try:
            self.llm = await self.model_router.get_engine_for_task("general_chat")
        except Exception as e:
            logger.warning("Primary LLM engine unavailable: %s", e)
            
            # Attempt recovery for LLM failure
            recovered = await recovery_manager.attempt_recovery(e, {"task": "initialize_llm"})
            if recovered:
                 self.llm = await self.model_router.get_engine_for_task("general_chat")
            else:
                 self.llm = None

    def _load_skills(self):
        """Initialize and register skills as MCP tools."""
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
        if self.model_router is None:
            return

        self.router = AgentRouter(self.model_router, self.config)

        # 1. Register Adaptive RAG
        mm = self.memory_manager
        if mm.vector_store and mm.document_indexer:
            retriever = LocalDocumentRetriever(mm.vector_store, mm.document_indexer)
            self.router.register_agent(AdaptiveRAGAgent(self.model_router, retriever, config=self.config))

        # 2. Register Code Assistant
        self.router.register_agent(
            CodeAssistantAgent(self.model_router, SandboxExecutor(self.config), config=self.config)
        )

        # 3. Register System Command Agent
        self.router.register_agent(SystemCommandAgent(self.model_router, config=self.config))

        # 4. Register Research Agent from plugins
        try:
            from ..plugins.research.main import ResearchAgent
            if mm.vector_store:
                self.router.register_agent(ResearchAgent(self.model_router, mm.vector_store, config=self.config))
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
        
        # Wait for AgentRunner's own background tasks
        if self._pending_tasks:
            logger.info(f"Waiting for {len(self._pending_tasks)} pending AgentRunner tasks...")
            try:
                # Use wait with timeout to avoid hanging indefinitely
                done, pending = await asyncio.wait(self._pending_tasks, timeout=5.0)
                if pending:
                    logger.warning(f"{len(pending)} tasks did not complete within timeout and will be cancelled.")
                    for task in pending:
                        task.cancel()
            except Exception as e:
                logger.error(f"Error while waiting for background tasks: {e}")
            self._pending_tasks.clear()

        if self.memory_manager.memory_consolidator and self.session.session_id:
            logger.info(f"Consolidating memory for session {self.session.session_id} before shutdown...")
            try:
                await self.memory_manager.consolidate_session(self.session.session_id)
            except Exception as e:
                logger.error(f"Memory consolidation failed during shutdown: {e}")

        if self.model_router:
            await self.model_router.aclose()
        
        if hasattr(self, "tts") and self.tts:
            await self.tts.aclose()
        
        await self.orchestrator.shutdown()

        logger.info("AgentRunner resources closed.")

    async def _add_to_history(self, role: str, content: Optional[str] = None, **kwargs) -> None:
        """Add a message to both session history and persistent storage."""
        # 1. In-memory session history
        if hasattr(self, "session"):
            self.session.add_message(role, content, **kwargs)
        
        # 2. Persistent SQLite history
        await self.memory_manager.add_to_history(self.session.session_id, role, content, **kwargs)

    async def _remember_exchange(self, user_text: str, assistant_text: str) -> None:
        """Persist completed exchanges for future retrieval (Vector Store only)."""
        await self.memory_manager.remember_exchange(
            self.session.session_id, 
            len(self.session.history), 
            user_text, 
            assistant_text
        )

    async def _try_execute_command(self, text: str) -> Optional[str]:
        """Attempt to find and execute a deterministic command handler."""
        handler_data = registry.find_handler(text)
        if not handler_data:
            return None

        cmd, match = handler_data
        logger.info(f"Executing command: {cmd.name}")
        trace_manager.add_event("command_matched", {"command": cmd.name}, f"Matched deterministic command: {cmd.name}")
        
        try:
            # Pass session and regex matches to the handler
            result = await cmd.handler(
                self.session,
                *match.groups(),
                llm=self.llm,
                config=self.config,
                tts=self.tts,
                vector_store=self.memory_manager.vector_store,
                conversation_memory=self.memory_manager.conversation_memory,
            )
            result_str = str(result)
            await self._add_to_history("assistant", result_str)
            await self._remember_exchange(text, result_str)
            trace_manager.end_trace(result_str, success=True)
            return result_str
        except Exception as e:
            logger.error(f"Command {cmd.name} failed: {e}", exc_info=True)
            
            # Attempt recovery for command failure
            recovered = await recovery_manager.attempt_recovery(e, {"command": cmd.name})
            if recovered:
                return f"I've attempted to fix the issue: {str(e)}. Please try your command again."

            error_msg = f"I encountered an error running '{cmd.name}': {str(e)}"
            trace_manager.end_trace(error_msg, success=False)
            return error_msg

    async def _route_to_agent(self, text: str, intent: Optional[str] = None) -> Optional[str]:
        """Classify intent and delegate to a specialized agent if appropriate."""
        if not self.router:
            return None

        intent_name = intent or "unknown"
        try:
            # Ask the router to classify the intent if not provided
            if not intent:
                intent_name = await self.router.detect_intent(text, self.session.history)
            
            trace_manager.add_event("intent_detected", {"intent": intent_name}, f"Router detected intent: {intent_name}")
            
            if intent_name in self.router._agents:
                logger.info(f"Routing to specialized agent: {intent_name}")
                agent_result = await self.router.route_to(intent_name, text, self.session.history)
                
                self._last_tts_content = agent_result.metadata.tts_content
                await self._add_to_history("assistant", agent_result.content)
                await self._remember_exchange(text, agent_result.content)
                trace_manager.end_trace(agent_result.content, success=agent_result.success)
                return agent_result.content
        except ProviderRateLimitError as e:
            logger.error(f"Agent routing failed due to rate limit: {e}")
            trace_manager.add_event("routing_failed", {"error": str(e), "type": "rate_limit"})
            return "I'm sorry, I've hit my usage limits for the API. Please try again later."
        except Exception as e:
            logger.error(f"Agent routing failed: {e}")
            
            # Attempt recovery for agent failure
            recovered = await recovery_manager.attempt_recovery(e, {"intent": intent_name})
            if not recovered:
                 trace_manager.add_event("routing_failed", {"error": str(e)})
            
        return None

    async def handle_input(self, text: str) -> AsyncIterator[str]:
        """Main entry point for processing any user input with streaming support."""
        text = text.strip()
        self._last_tts_content = None # Reset for each interaction
        if not text:
            yield "I'm listening, but I didn't hear anything."
            return

        # Ensure primary LLM is ready
        await self._initialize_primary_llm()

        # Start Trace
        trace_manager.start_trace(text, session_id=self.session.session_id)

        # Start background tasks to reduce TTFT
        memory_task = asyncio.create_task(self.memory_manager.build_memory_message(text, self._pending_tasks))
        self._pending_tasks.add(memory_task)
        memory_task.add_done_callback(self._pending_tasks.discard)

        # Parallelize intent detection with memory retrieval and history persistence
        intent_task = None
        if self.router:
            intent_task = asyncio.create_task(self.router.detect_intent(text, self.session.history))
            self._pending_tasks.add(intent_task)
            intent_task.add_done_callback(self._pending_tasks.discard)

        try:
            # 1. Standardized: Add user message to history
            await self._add_to_history("user", text)

            # 2. Check the Command Registry (Deterministic Logic)
            command_result = await self._try_execute_command(text)
            if command_result:
                if intent_task: intent_task.cancel()
                memory_task.cancel()
                yield command_result
                return

            # 3. Check for Specialized Agent Intent (AI Routing)
            intent = None
            if intent_task:
                try:
                    intent = await intent_task
                except Exception as e:
                    logger.warning(f"Intent detection task failed: {e}")

            agent_result = await self._route_to_agent(text, intent=intent)
            if agent_result:
                memory_task.cancel()
                yield agent_result
                return

            # 4. Fallback to LLM if no command/agent matches
            trace_manager.add_event("llm_fallback_started")
            
            full_response = ""
            async for chunk in self._fallback_to_llm(text, memory_task=memory_task):
                full_response += chunk
                yield chunk
                
            trace_manager.end_trace(full_response, success=True)
        finally:
            # Cleanup background tasks
            for task in [memory_task, intent_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

            # Post-interaction summarization to avoid resource contention
            try:
                summ_llm = await self.model_router.get_engine_for_task("summarization")
                task = asyncio.create_task(self.session.summarize(summ_llm))
                self.session._pending_tasks.add(task)
                task.add_done_callback(self.session._pending_tasks.discard)
            except Exception as e:
                logger.warning(f"Failed to trigger summarization: {e}")

    @property
    def last_tts_content(self) -> Optional[str]:
        """Returns the voice-friendly summary of the last agent response, if any."""
        return getattr(self, "_last_tts_content", None)

    async def _fallback_to_llm(self, text: str, memory_task: Optional[asyncio.Task] = None) -> AsyncIterator[str]:
        """Use the local LLM when no specific command is triggered, with streaming support."""
        # Ensure MCP servers are started
        await self.orchestrator.ensure_mcp_ready(self.skills)
        
        logger.info("No command match. Falling back to LLM.")
        if self.llm is None or not self.llm.is_available():
            engine_type = self.config.get("llm.engine", "ollama")
            if engine_type == "openai":
                yield "The API engine is unavailable. Please check your API key in config.yaml."
                return
            yield "The local LLM engine is unavailable. Install the required dependencies and start Ollama to enable free-form chat."
            return
        
        # Build chat history for LLM
        messages = self.session.build_llm_messages()
        
        # Await the pre-fetched memory task or trigger it now if missing
        if memory_task:
            try:
                memory_message = await memory_task
            except Exception as e:
                logger.warning(f"Background memory retrieval failed: {e}")
                memory_message = None
        else:
            memory_message = await self.memory_manager.build_memory_message(text, self._pending_tasks)
            
        if memory_message is not None:
            insert_at = 1 if messages and messages[0].role == "system" else 0
            messages.insert(insert_at, memory_message)
        
        tools = mcp_client.get_tools_for_llm() or None
            
        try:
            async for chunk in self._run_react_loop(text, messages, tools, stream=True):
                yield chunk
        except ProviderRateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            yield "I'm sorry, I've hit my usage limits for the API. Please try again later."
        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
            yield f"I'm sorry, my brain is feeling a bit foggy: {str(e)}"

    async def _run_react_loop(self, original_text: str, messages: List[Message], tools: Optional[List[Dict[str, Any]]], stream: bool = False) -> AsyncIterator[str]:
        """Executes the iterative tool-calling loop with optional streaming."""
        max_iterations = 5
        
        for _ in range(max_iterations):
            response = await self.llm.chat(messages, tools=tools, stream=stream)
            
            if not isinstance(response, AsyncIterator):
                # Non-streaming (either stream=False or it's a tool call)
                if not response.tool_calls:
                    # Final answer (non-streaming)
                    await self._add_to_history("assistant", response.content)
                    await self._remember_exchange(original_text, response.content)
                    yield response.content
                    return

                # Handle tool calls
                assistant_msg = Message(role="assistant", content=response.content, tool_calls=response.tool_calls)
                messages.append(assistant_msg)
                await self._add_to_history(**assistant_msg.model_dump(exclude_none=True))
                
                # Execute tools
                tool_executor = ToolExecutor(mcp_client, permission_manager=self.permission_manager)
                tool_results = await tool_executor.execute_tool_calls(response.tool_calls)
                
                for tool_msg in tool_results:
                    messages.append(tool_msg)
                    await self._add_to_history(**tool_msg.model_dump(exclude_none=True))
                # Continue loop to next iteration
            else:
                # Streaming final answer or tool calls
                full_content = ""
                collected_tool_calls = None
                
                async for chunk in response:
                    if chunk.content:
                        full_content += chunk.content
                        yield chunk.content
                    if chunk.tool_calls:
                        collected_tool_calls = chunk.tool_calls
                
                if not collected_tool_calls:
                    await self._add_to_history("assistant", full_content)
                    await self._remember_exchange(original_text, full_content)
                    return
                
                # Handle tool calls that appeared in the stream
                assistant_msg = Message(role="assistant", content=full_content, tool_calls=collected_tool_calls)
                messages.append(assistant_msg)
                await self._add_to_history(**assistant_msg.model_dump(exclude_none=True))
                
                # Execute tools
                tool_executor = ToolExecutor(mcp_client, permission_manager=self.permission_manager)
                tool_results = await tool_executor.execute_tool_calls(collected_tool_calls)
                
                for tool_msg in tool_results:
                    messages.append(tool_msg)
                    await self._add_to_history(**tool_msg.model_dump(exclude_none=True))
                # Continue loop to next iteration
        
        final_msg = "I've reached my thinking limit on this task."
        await self._add_to_history("assistant", final_msg)
        yield final_msg
