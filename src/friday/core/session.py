"""Session state management for FRIDAY."""

import logging
import json
import uuid
import asyncio
from typing import Optional, List, Dict, Any, Set

from ..llm.engine import Message, LLMEngine

logger = logging.getLogger(__name__)

_tiktoken_warned = False

def estimate_tokens(text: str) -> int:
    """Accurately estimate token count or fallback to a conservative heuristic."""
    global _tiktoken_warned
    try:
        import tiktoken
        # Use cl100k_base (GPT-4) as a good general-purpose proxy
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except (ImportError, Exception):
        if not _tiktoken_warned:
            logger.warning("The 'tiktoken' package is not installed. Falling back to conservative token estimation.")
            _tiktoken_warned = True
        # Fallback to conservative heuristic: 1 token per 3 characters (safer than 1/4)
        return (len(text) + 2) // 3

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
        self._pending_archival: List[Dict[str, Any]] = []

    async def aclose(self) -> None:
        """Wait for all pending summarization tasks to complete."""
        if self._pending_tasks:
            logger.info(f"Waiting for {len(self._pending_tasks)} pending summarization tasks...")
            try:
                done, pending = await asyncio.wait(self._pending_tasks, timeout=5.0)
                if pending:
                    for task in pending:
                        task.cancel()
            except Exception:
                pass
            self._pending_tasks.clear()

    def add_message(self, role: str, content: Optional[str] = None, **kwargs):
        """Add a message to the session history."""
        msg = {"role": role, "content": content}
        msg.update(kwargs)
        self.history.append(msg)
        if len(self.history) > self.max_history_messages:
            overflow = len(self.history) - self.max_history_messages
            archived_messages = self.history[:overflow]
            self.history = self.history[overflow:]
            # Buffer for later summarization to avoid resource contention during active chat
            self._pending_archival.extend(archived_messages)

    async def summarize(self, llm: Optional['LLMEngine'] = None) -> None:
        """Process buffered messages for summarization."""
        if not self._pending_archival:
            return
            
        async with self._summarize_lock:
            if not self._pending_archival:
                return
                
            to_process = list(self._pending_archival)
            self._pending_archival = []
            
            if llm:
                await self._summarize_messages(to_process, llm)
            else:
                self._append_to_summary(to_process)

    async def _summarize_messages(self, archived_messages: List[Dict[str, Any]], llm: 'LLMEngine') -> None:
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

    def _append_to_summary(self, archived_messages: List[Dict[str, Any]]) -> None:
        """Condense evicted messages into a rolling JSON-formatted summary fallback."""
        try:
            data = json.loads(self.history_summary) if self.history_summary else {}
            if not isinstance(data, dict):
                data = {}
        except json.JSONDecodeError:
            # If not JSON, treat it as existing plain-text context
            data = {"general_context": self.history_summary}

        # Ensure schema
        if "user_preferences" not in data: data["user_preferences"] = []
        if "active_tasks" not in data: data["active_tasks"] = []
        if "general_context" not in data: data["general_context"] = ""

        context_lines = [data["general_context"]] if data["general_context"] else []
        for message in archived_messages:
            role = message['role'].capitalize()
            content = message.get('content') or ""
            if message.get('tool_calls'):
                content += f" [Calls tools: {', '.join(tc.get('function', {}).get('name', '') for tc in message['tool_calls'])}]"
            context_lines.append(f"{role}: {content}")
            
        combined_context = "\n".join(line for line in context_lines if line)
        data["general_context"] = combined_context[-self.summary_max_chars:]
        
        self.history_summary = json.dumps(data)

    def build_llm_messages(
        self, 
        system_prompt: Optional[str] = None,
        max_tokens: int = 4000,
        include_summary: bool = True
    ) -> List[Message]:
        """Build the message list for LLM interaction with summarized context and token budgeting."""
        return self.format_messages(
            history=self.history,
            summary=self.history_summary if include_summary else None,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            recent_limit=self.recent_messages
        )

    @staticmethod
    def format_messages(
        history: List[Dict[str, Any]],
        summary: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4000,
        recent_limit: int = 20
    ) -> List[Message]:
        """Centralized logic to convert history and summary into LLM Messages."""
        actual_system_prompt = system_prompt or "You are FRIDAY, a helpful, privacy-first local AI assistant. Answer the user's request directly or use tools if needed."
        
        system_msg = Message(role="system", content=actual_system_prompt)
        messages: List[Message] = [system_msg]
        
        current_tokens = estimate_tokens(system_msg.content)

        if summary:
            summary_content = f"Conversation summary from earlier in this session:\n{summary}"
            summary_tokens = estimate_tokens(summary_content)
            
            # Reserve space for summary (up to 40% of budget)
            if current_tokens + summary_tokens < max_tokens * 0.4:
                messages.append(Message(role="system", content=summary_content))
                current_tokens += summary_tokens

        recent_history = history[-recent_limit:] if recent_limit > 0 else []
        valid_fields = {"role", "content", "name", "tool_calls", "tool_call_id"}
        
        history_to_add = []
        for entry in reversed(recent_history):
            filtered_entry = {k: v for k, v in entry.items() if k in valid_fields}
            content = filtered_entry.get("content") or ""
            entry_tokens = estimate_tokens(content)
            
            if current_tokens + entry_tokens > max_tokens:
                break
                
            history_to_add.insert(0, Message(**filtered_entry))
            current_tokens += entry_tokens
            
        messages.extend(history_to_add)
        return messages
