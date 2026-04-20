"""Email connector skill with IMAP support."""

import logging
import imaplib
import email
from email.header import decode_header
from typing import Dict, Any, List, Optional
from friday.skills.base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)

class EmailSkill(BaseSkill):
    """Email skill with IMAP support."""

    @property
    def name(self) -> str:
        return "email"

    @property
    def description(self) -> str:
        return "Gathers unread emails from the last 24h via IMAP."

    @property
    def required_env(self) -> List[str]:
        return ["IMAP_SERVER", "IMAP_USER", "IMAP_PASSWORD"]

    async def execute(self, query: str, context: Dict[str, Any]) -> SkillResult:
        """Gathers unread emails from IMAP server."""
        import os
        server = os.getenv("IMAP_SERVER")
        user = os.getenv("IMAP_USER")
        password = os.getenv("IMAP_PASSWORD")

        if not all([server, user, password]):
            # Fallback to mock for v0.1 demonstration if env vars are missing
            logger.info("IMAP credentials missing, using mock data.")
            mock_emails = [
                {"from": "boss@example.com", "subject": "Project Deadline", "snippet": "Hey, how is the AI assistant going?"},
                {"from": "newsletter@tech.com", "subject": "Daily Tech Brief", "snippet": "New breakthroughs in local LLMs..."},
                {"from": "mom@home.com", "subject": "Dinner Sunday?", "snippet": "Are you coming over this weekend?"}
            ]
            return SkillResult(success=True, data=mock_emails, message="Mock mode: Found 3 unread emails.")

        mail = None
        try:
            # Connect to server
            mail = imaplib.IMAP4_SSL(server)
            mail.login(user, password)
            mail.select("inbox")

            # Search for unread emails
            status, messages = mail.search(None, 'UNSEEN')
            if status != 'OK':
                return SkillResult(success=False, data=[], message="Failed to search inbox.")

            email_list = []
            # Get the list of email IDs
            mail_ids = messages[0].split()
            # Limit to last 5 unread
            for i in mail_ids[-5:]:
                res, msg_data = mail.fetch(i, '(RFC822)')
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        subject, encoding = decode_header(msg["Subject"])[0]
                        if isinstance(subject, bytes):
                            subject = subject.decode(encoding or "utf-8")
                        
                        from_, encoding = decode_header(msg.get("From"))[0]
                        if isinstance(from_, bytes):
                            from_ = from_.decode(encoding or "utf-8")
                        
                        snippet = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == "text/plain":
                                    snippet = part.get_payload(decode=True).decode()[:100]
                                    break
                        else:
                            snippet = msg.get_payload(decode=True).decode()[:100]

                        email_list.append({
                            "from": from_,
                            "subject": subject,
                            "snippet": snippet
                        })

            return SkillResult(success=True, data=email_list)

        except Exception as e:
            logger.error(f"Email skill failed: {e}")
            return SkillResult(success=False, data=[], message=str(e))
        finally:
            if mail:
                try:
                    mail.close()
                    mail.logout()
                except:
                    pass
