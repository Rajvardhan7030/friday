"""
Adaptive RAG Agent for FRIDAY.
Implements a LangGraph-based state machine for intelligent document retrieval.
Includes voice-friendly auto-summarization for TTS.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, TypedDict

from .base import BaseAgent, Context, AgentResult
from ..llm.engine import LLMEngine, Message
from .tools import LocalDocumentRetriever

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """The state object passed between nodes in the graph."""
    query: str
    chat_history: List[Dict[str, str]]
    needs_retrieval: bool
    documents: List[Dict[str, Any]]
    relevant_docs: List[Dict[str, Any]]
    answer: Optional[str]
    tts_answer: Optional[str]
    sources: List[str]
    iteration: int
    max_iterations: int

class AdaptiveRAGAgent(BaseAgent):
    """
    An intelligent RAG agent that self-corrects and summarizes for voice output.
    """

    def __init__(
        self, 
        llm_engine: LLMEngine, 
        retriever: LocalDocumentRetriever,
        max_retries: int = 2
    ):
        super().__init__(llm_engine)
        self.retriever = retriever
        self.max_retries = max_retries

    @property
    def name(self) -> str:
        return "adaptive_rag"

    @property
    def description(self) -> str:
        return "Intelligent document retrieval agent with voice-friendly summarization."

    async def run(self, ctx: Context) -> AgentResult:
        """Execute the Adaptive RAG workflow."""
        state: AgentState = {
            "query": ctx.user_query,
            "chat_history": ctx.chat_history,
            "needs_retrieval": False,
            "documents": [],
            "relevant_docs": [],
            "answer": None,
            "tts_answer": None,
            "sources": [],
            "iteration": 0,
            "max_iterations": self.max_retries
        }

        # 1. Analyze Intent
        state = await self._analyze_query(state)
        if not state["needs_retrieval"]:
            return await self._run_general_chat(ctx)

        # 2. Retrieval Loop
        while state["iteration"] <= state["max_iterations"]:
            # Retrieve
            state["documents"] = await self.retriever.retrieve(state["query"])
            
            # Grade
            state = await self._grade_documents(state)
            
            if state["relevant_docs"]:
                # Generate
                state = await self._generate(state)
                # Summarize for TTS if needed
                state = await self._summarize_for_voice(state)
                break
            else:
                # Transform Query
                if state["iteration"] < state["max_iterations"]:
                    state = await self._transform_query(state)
                    state["iteration"] += 1
                else:
                    break

        return self._format_final_result(state)

    async def _analyze_query(self, state: AgentState) -> AgentState:
        """Determines if the query requires document retrieval."""
        prompt = f"Analyze if this query needs looking up personal documents: \"{state['query']}\"\nRespond ONLY with JSON: {{\"needs_retrieval\": true/false}}"
        try:
            res = await self.llm.chat([Message(role="user", content=prompt)])
            data = self._parse_json(res.content)
            state["needs_retrieval"] = data.get("needs_retrieval", True)
        except Exception:
            state["needs_retrieval"] = True
        return state

    async def _grade_documents(self, state: AgentState) -> AgentState:
        """Filters retrieved chunks for relevance."""
        relevant_docs = []
        for doc in state["documents"]:
            prompt = f"Is this document relevant to '{state['query']}'?\nDoc: {doc['content'][:200]}\nRespond JSON: {{\"relevant\": true/false}}"
            try:
                res = await self.llm.chat([Message(role="user", content=prompt)])
                data = self._parse_json(res.content)
                if data.get("relevant", False):
                    relevant_docs.append(doc)
            except Exception:
                continue
        state["relevant_docs"] = relevant_docs
        return state

    async def _transform_query(self, state: AgentState) -> AgentState:
        """Rewrites the query for better retrieval."""
        prompt = f"Rewrite this query for better document retrieval: '{state['query']}'"
        res = await self.llm.chat([Message(role="user", content=prompt)])
        state["query"] = res.content.strip()
        return state

    async def _generate(self, state: AgentState) -> AgentState:
        """Produces the full answer with citations."""
        context = "\n".join([d['content'] for d in state["relevant_docs"]])
        prompt = f"Answer using this context. Cite sources like [source: file.md].\nContext: {context}\nQuery: {state['query']}"
        res = await self.llm.chat([Message(role="user", content=prompt)])
        state["answer"] = res.content
        state["sources"] = list(set([d['metadata'].get('source') for d in state["relevant_docs"] if d['metadata'].get('source')]))
        return state

    async def _summarize_for_voice(self, state: AgentState) -> AgentState:
        """Summarizes the answer for TTS if it's too long."""
        answer = state["answer"] or ""
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        
        if len(sentences) > 3:
            logger.info("Answer is long, generating voice summary.")
            prompt = f"Summarize this answer in exactly 2 friendly sentences for voice output:\n{answer}"
            res = await self.llm.chat([Message(role="user", content=prompt)])
            state["tts_answer"] = res.content.strip()
        else:
            state["tts_answer"] = answer
        return state

    def _format_final_result(self, state: AgentState) -> AgentResult:
        if not state["answer"]:
            return AgentResult(content="I couldn't find relevant info.", success=False)
        
        return AgentResult(
            content=state["answer"],
            metadata={
                "tts_content": state["tts_answer"],
                "sources": state["sources"],
                "retrieval_used": True
            }
        )

    async def _run_general_chat(self, ctx: Context) -> AgentResult:
        res = await self.llm.chat([Message(role="user", content=ctx.user_query)])
        return AgentResult(content=res.content, metadata={"tts_content": res.content, "retrieval_used": False})

    def _parse_json(self, text: str) -> Dict[str, Any]:
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group()) if match else {}
        except: return {}
