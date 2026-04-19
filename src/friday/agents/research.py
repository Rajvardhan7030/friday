"""Research Agent with ReAct loop and citations."""

import logging
import json
from typing import Dict, Any, List, Optional
from src.friday.agents.base import BaseAgent, Context, AgentResult
from src.friday.llm.engine import LLMEngine, Message
from src.friday.skills.web_search_skill import WebSearchSkill
from src.friday.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """Multi-hop research agent with citations and ReAct loop."""

    def __init__(self, llm_engine: LLMEngine, vector_store: VectorStore):
        super().__init__(llm_engine)
        self.vector_store = vector_store
        self.web_search = WebSearchSkill()

    @property
    def name(self) -> str:
        return "research"

    @property
    def description(self) -> str:
        return "Deep research agent that searches local memory and the web with citations."

    async def run(self, ctx: Context) -> AgentResult:
        """Run the research loop."""
        logger.info(f"Starting research on: {ctx.user_query}")
        
        # 1. Search local memory
        local_results = await self.vector_store.similarity_search(ctx.user_query, k=5)
        
        # 2. Search web
        web_results_skill = await self.web_search.execute(ctx.user_query, {})
        web_results = web_results_skill.data if web_results_skill.success else []
        
        # 3. Consolidate results for synthesis
        consolidated_context = self._consolidate_context(local_results, web_results)
        
        # 4. Generate final response with citations
        prompt = self._build_research_prompt(ctx.user_query, consolidated_context)
        
        messages = [
            Message(role="system", content="You are Friday Research Agent. You provide detailed answers with citations. Use [source: <filename>] for local files and [source: web: <url>] for web results."),
            Message(role="user", content=prompt)
        ]
        
        response = await self.llm.chat(messages)
        
        return AgentResult(
            content=response.content,
            citations=self._extract_citations(local_results, web_results)
        )

    def _consolidate_context(self, local: List[Dict], web: List[Dict]) -> str:
        """Merge local and web results into a single context string."""
        context = "--- LOCAL MEMORY ---\n"
        for i, res in enumerate(local):
            source = res["metadata"].get("source", f"memory_{i}")
            context += f"SOURCE: {source}\nCONTENT: {res['content']}\n\n"
            
        context += "--- WEB SEARCH ---\n"
        for i, res in enumerate(web):
            source = res.get("href", f"web_{i}")
            context += f"SOURCE: web: {source}\nCONTENT: {res.get('body', '')}\n\n"
            
        return context

    def _build_research_prompt(self, query: str, context: str) -> str:
        """Construct prompt for research synthesis."""
        return f"Research query: {query}\n\nBased on the following context, please provide a comprehensive answer with inline citations:\n\n{context}"

    def _extract_citations(self, local: List[Dict], web: List[Dict]) -> List[Dict[str, str]]:
        """List citations for the metadata."""
        citations = []
        for res in local:
            citations.append({"type": "local", "source": res["metadata"].get("source", "unknown")})
        for res in web:
            citations.append({"type": "web", "source": res.get("href", "unknown")})
        return citations
