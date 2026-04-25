"""
Meeting Preparation Agent for FRIDAY.
Gathers context from local documents and builds a comprehensive meeting briefing.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from .base import BaseAgent, Context, AgentResult
from ..llm.engine import LLMEngine, Message
from ..memory.vector_store import VectorStore

logger = logging.getLogger(__name__)

class MeetingPrepCrew(BaseAgent):
    """
    A crew-based agent that scouts documents, builds agendas, and writes briefings.
    """

    def __init__(self, llm_engine: LLMEngine, vector_store: VectorStore):
        super().__init__(llm_engine)
        self.vector_store = vector_store
        self.briefings_dir = Path.home() / ".friday" / "briefings"
        self.briefings_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str: return "meeting_prep"

    @property
    def description(self) -> str:
        return "Prepares you for upcoming meetings by analyzing related documents and drafting agendas."

    async def run(self, ctx: Context) -> AgentResult:
        """Execute the meeting preparation workflow."""
        # 1. Document Scout: Find relevant files
        print(f"🔍 [Scout]: Searching for files related to: {ctx.user_query}...")
        docs = await self.vector_store.similarity_search(ctx.user_query, k=5)
        context_text = "\n\n".join([d['content'] for d in docs])
        
        # 2. Agenda Builder: Draft talking points
        print(f"📝 [Builder]: Drafting agenda and key talking points...")
        agenda_prompt = f"Based on these documents, draft a 3-bullet agenda for: {ctx.user_query}\n\nContext:\n{context_text}"
        agenda_res = await self.llm.chat([Message(role="user", content=agenda_prompt)])
        
        # 3. Briefing Writer: Compile final report
        print(f"✍️ [Writer]: Finalizing the markdown briefing...")
        writer_prompt = f"""Create a markdown briefing for a meeting: {ctx.user_query}.
Include:
- Meeting Purpose
- Key Documents (from context)
- 3 Suggested Questions
- Follow-up Items

Context:
{context_text}
Agenda:
{agenda_res.content}"""
        
        final_res = await self.llm.chat([Message(role="user", content=writer_prompt)])
        briefing_md = final_res.content
        
        # Save to file
        filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_briefing.md"
        file_path = self.briefings_dir / filename
        with open(file_path, "w") as f:
            f.write(briefing_md)
            
        # Voice Summary
        voice_prompt = f"Summarize this meeting briefing in 2 sentences for voice output:\n{briefing_md}"
        voice_res = await self.llm.chat([Message(role="user", content=voice_prompt)])
        
        return AgentResult(
            content=f"Briefing saved to {file_path}\n\n{briefing_md}",
            metadata={"tts_content": voice_res.content, "file_path": str(file_path)}
        )
