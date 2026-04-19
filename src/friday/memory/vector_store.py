"""Local vector storage using ChromaDB."""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from friday.llm.engine import LLMEngine

logger = logging.getLogger(__name__)

class VectorStore:
    """Wrapper around ChromaDB for local vector storage."""

    def __init__(self, persist_directory: str, llm_engine: LLMEngine):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(allow_reset=True)
        )
        self.llm = llm_engine
        self.collection = self.client.get_or_create_collection(
            name="friday_memory",
            metadata={"hnsw:space": "cosine"}
        )

    async def add_documents(
        self, 
        documents: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to the vector store with embeddings."""
        try:
            # Generate embeddings in batch if possible, but for now one by one
            embeddings = []
            for doc in documents:
                emb = await self.llm.embed(doc)
                embeddings.append(emb)

            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids or [f"doc_{i}" for i in range(len(documents))]
            )
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")

    async def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            query_embedding = await self.llm.embed(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter
            )
            
            # Format results
            formatted_results = []
            if results["documents"]:
                for i in range(len(results["documents"][0])):
                    formatted_results.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0
                    })
            return formatted_results
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
