"""Research Agent with ReAct loop and citations."""

import logging
from typing import Dict, List
from .base import BaseAgent, Context, AgentResult
from ..llm.engine import LLMEngine, Message
from ..skills.web_search_skill import WebSearchSkill
from ..memory.vector_store import VectorStore

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    """Multi-hop research agent with citations and ReAct loop."""

    def __init__(self, llm_engine: LLMEngine, vector_store: VectorStore, max_iterations: int = 5):
        super().__init__(llm_engine)
        self.vector_store = vector_store
        self.web_search = WebSearchSkill()
        self.max_iterations = max_iterations

    @property
    def name(self) -> str:
        return "research"

    @property
    def description(self) -> str:
        return "Deep research agent that searches local memory and the web with citations."

    async def run(self, ctx: Context) -> AgentResult:
        """Run the research loop with a hard iteration limit."""
        logger.info(f"Starting research on: {ctx.user_query}")
        
        current_step = 0
        local_results = []
        web_results = []
        current_query = ctx.user_query
        
        while current_step < self.max_iterations:
            current_step += 1
            logger.debug(f"Research step {current_step}/{self.max_iterations} with query: {current_query}")
            
            # 1. Search local memory
            step_local = await self.vector_store.similarity_search(current_query, k=3)
            
            # Early termination: if we find highly relevant local content, maybe we don't need web?
            # For simplicity, we'll check if any result has very high similarity (if vector_store provided it)
            # Since our vector_store doesn't return scores yet, we'll just collect them.
            local_results.extend(step_local)
            
            # 2. Search web if needed (e.g., if local results are insufficient or we want to expand)
            # In a real ReAct loop, the LLM would decide if it needs more info.
            # Here we'll skip web search if we have enough local info (mock check)
            if not step_local or current_step > 1:
                web_results_skill = await self.web_search.execute(current_query, {})
                if web_results_skill.success:
                    web_results.extend(web_results_skill.data)
            
            # 3. Simple multi-hop: ask LLM if we need more info or if we can answer
            # To keep it efficient, we only do this if max_iterations > 1
            if self.max_iterations > 1 and current_step < self.max_iterations:
                refine_prompt = f"Original query: {ctx.user_query}\nCurrent results: {len(local_results)} local, {len(web_results)} web.\nBased on what we have, what is the next specific question to search for to provide a complete answer? Respond ONLY with the new search query or 'FINISH' if we have enough."
                refine_res = await self.llm.chat([Message(role="user", content=refine_prompt)])
                if "FINISH" in refine_res.content.upper():
                    break
                current_query = refine_res.content.strip().strip('"').strip("'")
            else:
                break
            
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
        # Deduplicate by source
        seen_sources = set()
        for i, res in enumerate(local):
            source = res["metadata"].get("source", f"memory_{i}")
            if source not in seen_sources:
                context += f"SOURCE: {source}\nCONTENT: {res['content']}\n\n"
                seen_sources.add(source)
            
        context += "--- WEB SEARCH ---\n"
        for i, res in enumerate(web):
            source = res.get("href", f"web_{i}")
            if source not in seen_sources:
                # Basic HTML sanitization for prompt safety
                body = res.get('body', '')
                clean_body = self._strip_html(body)
                context += f"SOURCE: web: {source}\nCONTENT: {clean_body}\n\n"
                seen_sources.add(source)
            
        return context

    def _strip_html(self, text: str) -> str:
        """Very basic HTML tag stripper."""
        import re
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def _build_research_prompt(self, query: str, context: str) -> str:
        """Construct prompt for research synthesis."""
        return f"Research query: {query}\n\nBased on the following context, please provide a comprehensive answer with inline citations:\n\n{context}"

    def _extract_citations(self, local: List[Dict], web: List[Dict]) -> List[Dict[str, str]]:
        """List citations for the metadata."""
        citations = []
        seen = set()
        for res in local:
            source = res["metadata"].get("source", "unknown")
            if source not in seen:
                citations.append({"type": "local", "source": source})
                seen.add(source)
        for res in web:
            source = res.get("href", "unknown")
            if source not in seen:
                citations.append({"type": "web", "source": source})
                seen.add(source)
        return citations
