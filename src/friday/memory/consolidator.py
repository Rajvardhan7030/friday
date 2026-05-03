"""
Memory Consolidator for FRIDAY.
Handles promotion of facts from STM (Conversation) to LTM (Vector Store).
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional
from .conversation import ConversationMemory
from .vector_store import VectorStore
from ..llm.engine import LLMEngine, Message

logger = logging.getLogger(__name__)

class MemoryConsolidator:
    """
    Consolidates transient conversation history into long-term factual memory.
    """

    def __init__(
        self, 
        llm: LLMEngine, 
        vector_store: VectorStore,
        conversation_memory: Optional[ConversationMemory] = None,
        ltm_collection: str = "ltm_memory"
    ):
        self.llm = llm
        self.vector_store = vector_store
        self.conversation_memory = conversation_memory
        self.ltm_collection = ltm_collection

    async def consolidate_session(self, session_id: str, limit: int = 20) -> int:
        """
        Extract facts from a session's history and promote them to LTM.
        Returns the number of facts promoted.
        """
        if not self.conversation_memory:
            logger.warning("ConversationMemory not provided, cannot consolidate session.")
            return 0

        # 1. Retrieve STM
        history = await self.conversation_memory.get_history(session_id, limit=limit)
        if not history:
            return 0

        # 2. Extract Facts via LLM
        facts = await self._extract_facts(history)
        if not facts:
            return 0

        # 3. Store in LTM
        await self.vector_store.add_documents(
            documents=facts,
            metadatas=[{"source": "conversation_consolidation", "session_id": session_id} for _ in facts],
            collection_name=self.ltm_collection
        )
        
        logger.info(f"Promoted {len(facts)} facts to LTM for session {session_id}")
        return len(facts)

    async def _extract_facts(self, history: List[Dict[str, str]]) -> List[str]:
        """Use the LLM to identify permanent facts from the chat history."""
        chat_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history])
        
        prompt = f"""
Analyze the following conversation and extract up to 3 permanent, factual insights about the user or their preferences.
Focus on durable information (e.g., "User is a React developer", "User prefers dark mode", "User's dog is named Max").
Ignore transient task context or greetings.

Conversation:
{chat_text}

Respond ONLY with a JSON list of strings. If no permanent facts are found, return [].
Example: ["User works as a software engineer", "User lives in New York"]
"""
        
        try:
            response = await self.llm.chat([Message(role="system", content=prompt)])
            content = response.content.strip()
            
            # Robust JSON extraction: Find the first '[' and last ']'
            start_idx = content.find('[')
            end_idx = content.rfind(']')
            
            facts = []
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx + 1]
                try:
                    facts = json.loads(json_str)
                except json.JSONDecodeError:
                    # Fallback to regex if nested structures or additional text cause issues
                    match = re.search(r"(\[.*\])", content, re.DOTALL)
                    if match:
                        try:
                            facts = json.loads(match.group(1))
                        except json.JSONDecodeError:
                            facts = []
            else:
                try:
                    facts = json.loads(content)
                except json.JSONDecodeError:
                    facts = []
                
            if isinstance(facts, list):
                return [str(f) for f in facts if f]
        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
            
        return []
