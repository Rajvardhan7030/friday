"""
Tools for FRIDAY Agents.
Includes local document indexing and retrieval tools.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional
from ..memory.vector_store import VectorStore
from ..memory.document_indexer import DocumentIndexer

logger = logging.getLogger(__name__)

class LocalDocumentRetriever:
    """
    Scans ~/Documents and manages a local vector index for FRIDAY.
    """

    def __init__(self, vector_store: VectorStore, indexer: DocumentIndexer):
        self.vector_store = vector_store
        self.indexer = indexer
        self.documents_dir = Path.home() / "Documents"
        self._initialized = False

    async def ensure_indexed(self, force: bool = False):
        """
        Builds the index if it doesn't exist or if forced.
        """
        if self._initialized and not force:
            return

        # Check if collection is empty
        await self.vector_store.initialize()
        count = self.vector_store.collection.count()
        
        if count == 0 or force:
            logger.info(f"Indexing Documents directory: {self.documents_dir}")
            # Scan for .txt, .md, and .pdf (pdf requires extra handling in real world)
            # For this prototype, we use the supported extensions in DocumentIndexer
            total_chunks = await self.indexer.index_directory(self.documents_dir)
            logger.info(f"Indexing complete. Added {total_chunks} chunks.")
        
        self._initialized = True

    async def retrieve(self, query: str, k: int = 5) -> List[dict]:
        """
        Retrieves relevant documents for a query.
        """
        await self.ensure_indexed()
        return await self.vector_store.similarity_search(query, k=k)
