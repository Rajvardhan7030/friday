import aiosqlite
import chromadb
import time
import uuid
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConversationMemory:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    timestamp REAL,
                    role TEXT,
                    content TEXT,
                    platform TEXT
                )
            ''')
            await db.commit()

    async def add_message(self, conversation_id: str, role: str, content: str, platform: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                'INSERT INTO messages (conversation_id, timestamp, role, content, platform) VALUES (?, ?, ?, ?, ?)',
                (conversation_id, time.time(), role, content, platform)
            )
            await db.commit()
            
    async def get_messages(self, conversation_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                'SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC LIMIT ?',
                (conversation_id, limit)
            ) as cursor:
                rows = await cursor.fetchall()
                return [{"role": row[0], "content": row[1]} for row in rows]

class SemanticMemory:
    def __init__(self, persist_dir: str = "chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="conversations")
        
    async def remember(self, text: str, embedding: List[float], metadata: Dict[str, Any]):
        doc_id = str(uuid.uuid4())
        self.collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
    async def recall(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        recalls = []
        if results and "documents" in results and results["documents"]:
            for i in range(len(results["documents"][0])):
                recalls.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i]
                })
        return recalls
