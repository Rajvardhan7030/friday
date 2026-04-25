"""Local vector storage using ChromaDB."""

import logging
from typing import List, Dict, Any, Optional
from ..llm.engine import LLMEngine

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    Settings = None

logger = logging.getLogger(__name__)

class VectorStore:
    """Wrapper around ChromaDB for local vector storage."""

    def __init__(self, persist_directory: str, llm_engine: LLMEngine):
        self.persist_directory = persist_directory
        self.llm = llm_engine
        self.client = None
        self.collection = None

    async def initialize(self) -> None:
        """Explicitly initialize the ChromaDB client and collection."""
        if chromadb is None or Settings is None:
            raise RuntimeError("The 'chromadb' package is not installed. Install project dependencies to use vector storage.")
        if self.client is None:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(allow_reset=True)
            )
            self.collection = self.client.get_or_create_collection(
                name="friday_memory",
                metadata={"hnsw:space": "cosine"}
            )
        logger.info(f"VectorStore initialized at {self.persist_directory}")

    async def add_documents(
        self, 
        documents: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to the vector store with embeddings."""
        if self.collection is None:
            await self.initialize()

        if not documents:
            return

        try:
            # Generate embeddings - still one by one but more robust
            embeddings = []
            valid_docs = []
            valid_metadatas = []
            valid_ids = []
            
            for i, doc in enumerate(documents):
                emb = await self.llm.embed(doc)
                if not emb:
                    logger.warning(f"Failed to generate embedding for document: {doc[:50]}...")
                    continue
                
                embeddings.append(emb)
                valid_docs.append(doc)
                if metadatas:
                    valid_metadatas.append(metadatas[i])
                if ids:
                    valid_ids.append(ids[i])
                else:
                    valid_ids.append(f"doc_{len(valid_ids)}")

            if not embeddings:
                return

            self.collection.add(
                embeddings=embeddings,
                documents=valid_docs,
                metadatas=valid_metadatas if valid_metadatas else None,
                ids=valid_ids
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
        if self.collection is None:
            await self.initialize()

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
