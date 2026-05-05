"""SQLite-backed conversation history with automatic summarization."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
logger = logging.getLogger(__name__)

class ChatMessage(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), index=True)
    role = Column(String(20))
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON, nullable=True)

class ConversationMemory:
    """Manages chat history and summarization."""

    def __init__(self, db_path: str):
        # Enforce connection pooling and WAL mode for concurrency
        self.engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            connect_args={"timeout": 15.0} # Wait if locked
        )
        
        # Enable Write-Ahead Logging on connection
        @event.listens_for(self.engine.sync_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

        self.session_factory = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a message to history."""
        async with self.session_factory() as session:
            msg = ChatMessage(
                session_id=session_id,
                role=role,
                content=content,
                metadata_json=metadata
            )
            session.add(msg)
            await session.commit()

    async def get_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent chat history."""
        from sqlalchemy import select
        async with self.session_factory() as session:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.timestamp.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            messages = result.scalars().all()
            
            # Return in chronological order
            history = []
            for m in reversed(messages):
                entry = {"role": m.role, "content": m.content}
                if m.metadata_json:
                    entry.update(m.metadata_json)
                history.append(entry)
            return history

    async def clear_history(self, session_id: str) -> None:
        """Clear history for a session."""
        from sqlalchemy import delete
        async with self.session_factory() as session:
            stmt = delete(ChatMessage).where(ChatMessage.session_id == session_id)
            await session.execute(stmt)
            await session.commit()
