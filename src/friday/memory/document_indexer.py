"""Document ingestion and indexing with semantic chunking."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import aiofiles
from friday.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)

class DocumentIndexer:
    """Handles chunking and indexing of local documents."""

    def __init__(self, vector_store: VectorStore, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def index_file(self, file_path: Path) -> int:
        """Read, chunk, and index a single file."""
        if not file_path.exists() or not file_path.is_file():
            logger.warning(f"File {file_path} not found.")
            return 0

        try:
            async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
                content = await f.read()

            chunks = self._chunk_text(content)
            metadatas = [{"source": str(file_path), "chunk": i} for i in range(len(chunks))]
            ids = [f"{file_path.name}_{i}" for i in range(len(chunks))]

            await self.vector_store.add_documents(chunks, metadatas, ids)
            return len(chunks)
        except Exception as e:
            logger.error(f"Failed to index file {file_path}: {e}")
            return 0

    async def index_directory(self, dir_path: Path, glob_pattern: str = "**/*.*") -> int:
        """Index all files matching a pattern in a directory."""
        if not dir_path.exists() or not dir_path.is_dir():
            logger.warning(f"Directory {dir_path} not found.")
            return 0

        total_chunks = 0
        # For simplicity, we only index common text formats for now
        allowed_extensions = {".txt", ".md", ".pdf", ".py", ".yaml", ".yml", ".json"}
        
        for file_path in dir_path.glob(glob_pattern):
            if file_path.suffix.lower() in allowed_extensions:
                chunks_added = await self.index_file(file_path)
                total_chunks += chunks_added
                logger.debug(f"Indexed {file_path}: {chunks_added} chunks added.")

        return total_chunks

    def _chunk_text(self, text: str) -> List[str]:
        """Simple sliding window chunking."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
            
            # Prevent infinite loop if overlap is larger than chunk size
            if start >= len(text) or self.chunk_overlap >= self.chunk_size:
                break
                
        return chunks
