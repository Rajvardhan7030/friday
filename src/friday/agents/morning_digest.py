"""Morning Digest Agent."""

import logging
from typing import Dict, Any, List
from friday.agents.base import BaseAgent, Context, AgentResult
from friday.llm.engine import LLMEngine, Message
from friday.skills.email_skill import EmailSkill
from friday.skills.calendar_skill import CalendarSkill
from friday.skills.news_skill import NewsSkill
from friday.voice.tts import TTSEngine

logger = logging.getLogger(__name__)

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
        """Run the morning briefing pipeline."""
        logger.info("Starting morning digest...")
        
        # 1. Gather data
        emails = await self.email_skill.execute("", {})
        calendar = await self.calendar_skill.execute("", {})
        news = await self.news_skill.execute("", {})
        
        # 2. Build prompt
        prompt = self._build_briefing_prompt(
            emails.data if emails.success else [],
            calendar.data if calendar.success else [],
            news.data if news.success else []
        )
        
        # 3. Generate briefing via LLM
        messages = [
            Message(role="system", content="You are Friday, a helpful and concise personal AI assistant. Your goal is to provide a clear and engaging morning briefing."),
            Message(role="user", content=prompt)
        ]
        
        response = await self.llm.chat(messages)
        briefing_text = response.content
        
        # 4. Speak briefing
        await self.tts.speak(briefing_text)
        
        return AgentResult(
            content=briefing_text,
            metadata={
                "emails_count": len(emails.data) if emails.success else 0,
                "events_count": len(calendar.data) if calendar.success else 0,
                "news_count": len(news.data) if news.success else 0
            }
        )

    def _build_briefing_prompt(self, emails: List[Dict], calendar: List[Dict], news: List[Dict]) -> str:
        """Construct the prompt for LLM synthesis."""
        prompt = "Please provide a concise morning briefing (max 400 tokens) based on the following information:\n\n"
        
        prompt += "--- UNREAD EMAILS ---\n"
        for email in emails:
            prompt += f"- From: {email['from']}, Subject: {email['subject']}\n"
            
        prompt += "\n--- CALENDAR EVENTS ---\n"
        for event in calendar:
            prompt += f"- {event['time']}: {event['event']}\n"
            
        prompt += "\n--- TOP NEWS HEADLINES ---\n"
        for item in news:
            prompt += f"- {item['title']}\n"
            
        prompt += "\nEnd of information. Please format your response as a friendly verbal briefing."
        return prompt
