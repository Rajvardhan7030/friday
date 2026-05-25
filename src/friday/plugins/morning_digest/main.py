"""Morning Digest Agent."""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List
from friday.agents.base import BaseAgent, Context, AgentResult, AgentMetadata
from friday.llm.engine import LLMEngine, Message
from friday.core.registry import registry
from friday.core.agent_runner import Session
from friday.core.config import Config
from friday.plugins.email.main import EmailSkill
from friday.plugins.calendar.main import CalendarSkill
from friday.plugins.news.main import NewsSkill
from friday.voice.tts import TTSEngine

logger = logging.getLogger(__name__)


@registry.register(
    name="Morning Digest",
    regex=r"^(?:morning digest|daily briefing|start (?:the )?morning digest|give me (?:my )?morning digest)$",
    description="Generate and read out your morning briefing",
    usage="morning digest",
    priority=8
)
async def morning_digest_handler(
    session: Session,
    llm: LLMEngine | None = None,
    config: Config | None = None,
    tts: TTSEngine | None = None,
    **_kwargs,
):
    """Entry point to run the morning digest from the command registry."""
    if llm is None:
        return "Internal Error: LLM engine not available for the morning digest."
    if config is None:
        return "Internal Error: Configuration not available for the morning digest."
    if tts is None:
        return "Internal Error: TTS engine not available for the morning digest."

    agent = MorningDigestAgent(llm, tts)
    result = await agent.run(
        Context(
            user_query="morning digest",
            chat_history=session.history,
        )
    )
    return result.content

class MorningDigestAgent(BaseAgent):
    """Gathers emails, calendar, and news for a morning briefing."""

    def __init__(self, llm_engine: LLMEngine, tts_engine: TTSEngine):
        super().__init__(llm_engine)
        self.tts = tts_engine
        self.email_skill = EmailSkill()
        self.calendar_skill = CalendarSkill()
        self.news_skill = NewsSkill()

    @property
    def name(self) -> str:
        return "morning_digest"

    @property
    def description(self) -> str:
        return "Gathers unread emails, calendar events, and news to generate a daily briefing."

    async def run(self, ctx: Context) -> AgentResult:
        """Run the morning briefing pipeline with timezone awareness."""
        now = datetime.now()
        current_time_str = now.strftime("%A, %B %d, %Y %I:%M %p")
        logger.info(f"Starting morning digest at {current_time_str}...")

        # 1. Gather data in parallel
        context = {"current_time": current_time_str, "date": now.strftime("%Y-%m-%d")}
        
        results = await asyncio.gather(
            self.email_skill.execute("", context),
            self.calendar_skill.execute("", context),
            self.news_skill.execute("", context),
            return_exceptions=True
        )
        
        # Handle potential exceptions in gather
        emails = results[0] if not isinstance(results[0], Exception) else type('obj', (object,), {'success': False, 'data': []})
        calendar = results[1] if not isinstance(results[1], Exception) else type('obj', (object,), {'success': False, 'data': []})
        news = results[2] if not isinstance(results[2], Exception) else type('obj', (object,), {'success': False, 'data': []})

        # 2. Build prompt
        prompt = self._build_briefing_prompt(
            current_time_str,
            emails.data if hasattr(emails, 'success') and emails.success else [],
            calendar.data if hasattr(calendar, 'success') and calendar.success else [],
            news.data if hasattr(news, 'success') and news.success else []
        )

        # 3. Generate briefing via LLM
        messages = [
            Message(role="system", content=f"You are Friday, a helpful and concise personal AI assistant. The current time is {current_time_str}. Your goal is to provide a clear and engaging morning briefing."),
            Message(role="user", content=prompt)
        ]

        response = await self.llm.chat(messages)
        briefing_text = response.content

        # 4. Speak briefing
        if self.tts:
            await self.tts.speak(briefing_text)

        return AgentResult(
            content=briefing_text,
            metadata=AgentMetadata(
                tts_content=briefing_text
            )
        )

    def _build_briefing_prompt(self, current_time: str, emails: List[Dict], calendar: List[Dict], news: List[Dict]) -> str:
        """Construct the prompt for LLM synthesis."""
        prompt = f"The current time is {current_time}. Please provide a concise morning briefing (max 400 tokens) based on the following information:\n\n"

        prompt += "--- UNREAD EMAILS ---\n"
        # Deduplication could be done here if needed
        seen_subjects = set()
        for email in emails:
            if email['subject'] not in seen_subjects:
                prompt += f"- From: {email['from']}, Subject: {email['subject']}\n"
                seen_subjects.add(email['subject'])

        prompt += "\n--- CALENDAR EVENTS ---\n"
        for event in calendar:
            prompt += f"- {event['time']}: {event['event']}\n"
            
        prompt += "\n--- TOP NEWS HEADLINES ---\n"
        for item in news:
            prompt += f"- {item['title']}\n"
            
        prompt += "\nEnd of information. Please format your response as a friendly verbal briefing."
        return prompt
