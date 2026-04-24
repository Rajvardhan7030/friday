"""Morning Digest Agent."""

import logging
from typing import Dict, List
from .base import BaseAgent, Context, AgentResult
from ..llm.engine import LLMEngine, Message
from ..core.registry import registry
from ..core.agent_runner import Session
from ..core.config import Config
from ..skills.email_skill import EmailSkill
from ..skills.calendar_skill import CalendarSkill
from ..skills.news_skill import NewsSkill
from ..voice.tts import TTSEngine

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
    **_kwargs,
):
    """Entry point to run the morning digest from the command registry."""
    if llm is None:
        return "Internal Error: LLM engine not available for the morning digest."
    if config is None:
        return "Internal Error: Configuration not available for the morning digest."

    agent = MorningDigestAgent(llm, TTSEngine(config))
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
        from datetime import datetime
        now = datetime.now()
        current_time_str = now.strftime("%A, %B %d, %Y %I:%M %p")
        logger.info(f"Starting morning digest at {current_time_str}...")

        # 1. Gather data (pass current date context)
        context = {"current_time": current_time_str, "date": now.strftime("%Y-%m-%d")}
        emails = await self.email_skill.execute("", context)
        calendar = await self.calendar_skill.execute("", context)
        news = await self.news_skill.execute("", context)

        # 2. Build prompt
        prompt = self._build_briefing_prompt(
            current_time_str,
            emails.data if emails.success else [],
            calendar.data if calendar.success else [],
            news.data if news.success else []
        )

        # 3. Generate briefing via LLM
        messages = [
            Message(role="system", content=f"You are Friday, a helpful and concise personal AI assistant. The current time is {current_time_str}. Your goal is to provide a clear and engaging morning briefing."),
            Message(role="user", content=prompt)
        ]

        response = await self.llm.chat(messages)
        briefing_text = response.content

        # 4. Speak briefing
        await self.tts.speak(briefing_text)

        return AgentResult(
            content=briefing_text,
            metadata={
                "current_time": current_time_str,
                "emails_count": len(emails.data) if emails.success else 0,
                "events_count": len(calendar.data) if calendar.success else 0,
                "news_count": len(news.data) if news.success else 0
            }
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
