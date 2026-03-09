"""벡터 저장소 인터페이스 — Repository 패턴."""
from abc import ABC, abstractmethod
from processing.chunker import Chunk


class BaseStore(ABC):
    """SQLite, ChromaDB, Pinecone 등으로 교체 가능. (DIP)"""

    @abstractmethod
    def save_document(self, doc_id: str, source: str, content: str,
                      doc_type: str, chunk_count: int) -> None: ...

    @abstractmethod
    def save_chunks(self, chunks: list[Chunk],
                    embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def search_similar(self, query_embedding: list[float],
                       top_k: int) -> list[dict]: ...

    @abstractmethod
    def get_stats(self) -> dict: ...
