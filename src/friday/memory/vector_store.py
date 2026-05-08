"""Local vector storage using ChromaDB."""

import logging
import uuid
import asyncio
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

    def __init__(self, persist_directory: str, llm_engine: LLMEngine, default_collection: str = "friday_memory"):
        self.persist_directory = persist_directory
        self.llm = llm_engine
        self.client = None
        self.collection = None
        self.default_collection = default_collection
        self._collections: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def initialize(self, collection_name: Optional[str] = None) -> None:
        """Explicitly initialize the ChromaDB client and collection."""
        if chromadb is None or Settings is None:
            raise RuntimeError("The 'chromadb' package is not installed. Install project dependencies to use vector storage.")
        
        async with self._lock:
            if self.client is None:
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(allow_reset=True)
                )
            
            name = collection_name or self.default_collection
            if name not in self._collections:
                self._collections[name] = self.client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"}
                )
            
            self.collection = self._collections[name]
        logger.info(f"VectorStore collection '{name}' initialized at {self.persist_directory}")

    async def get_collection(self, name: str) -> Any:
        """Get or create a specific collection."""
        async with self._lock:
            if self.client is None:
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(allow_reset=True)
                )
            
            if name not in self._collections:
                self._collections[name] = self.client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"}
                )
            return self._collections[name]

    async def add_documents(
        self, 
        documents: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None
    ) -> None:
        """Add documents to the vector store with embeddings."""
        target_collection = self.collection
        if collection_name:
            target_collection = await self.get_collection(collection_name)
        
        if target_collection is None:
            await self.initialize()
            target_collection = self.collection

        if not documents:
            return

        try:
            # Generate embeddings in batch
            embeddings = await self.llm.embed_batch(documents)
            
            if not embeddings:
                return

            valid_ids = ids if ids else [str(uuid.uuid4()) for _ in range(len(documents))]

            await asyncio.to_thread(
                target_collection.add,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas if metadatas else None,
                ids=valid_ids
            )
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")

    async def similarity_search(
        self, 
        query: str, 
        k: int = 5, 
        filter: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        query_embedding: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        target_collection = self.collection
        name = collection_name or self.default_collection
        if collection_name:
            target_collection = await self.get_collection(collection_name)

        if target_collection is None:
            await self.initialize(name)
            target_collection = self.collection

        try:
            if query_embedding is None:
                query_embedding = await self.llm.embed(query)
            
            results = await asyncio.to_thread(
                target_collection.query,
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
            err_msg = str(e)
            if "dimension" in err_msg.lower() and ("expected" in err_msg.lower() or "expecting" in err_msg.lower()):
                logger.warning(f"Embedding dimension mismatch in collection '{name}'. This usually happens when the embedding model is changed. Resetting collection to maintain compatibility.")
                await self.reset_collection(name)
            else:
                logger.error(f"Similarity search failed: {e}")
            return []

    async def reset_collection(self, collection_name: str) -> None:
        """Delete and recreate a collection."""
        async with self._lock:
            if self.client:
                try:
                    self.client.delete_collection(collection_name)
                    if collection_name in self._collections:
                        del self._collections[collection_name]
                    
                    # Re-create
                    self._collections[collection_name] = self.client.create_collection(
                        name=collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    if not self.default_collection or collection_name == self.default_collection:
                        self.collection = self._collections[collection_name]
                    
                    logger.info(f"Collection '{collection_name}' has been reset.")
                except Exception as e:
                    logger.error(f"Failed to reset collection '{collection_name}': {e}")
