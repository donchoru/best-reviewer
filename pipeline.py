"""RAG 파이프라인 — DI 기반 오케스트레이터."""
import hashlib
import logging
from config import RAGConfig
from loaders import BaseLoader
from processing import TextChunker, BaseEmbedder
from stores import BaseStore

logger = logging.getLogger(__name__)


class LoaderRegistry:
    """로더 레지스트리 — 새 로더를 런타임에 등록/조회. (OCP)"""

    def __init__(self):
        self._loaders: dict[str, BaseLoader] = {}

    def register(self, loader: BaseLoader) -> None:
        self._loaders[loader.source_type] = loader

    def get(self, source_type: str) -> BaseLoader:
        loader = self._loaders.get(source_type)
        if not loader:
            raise ValueError(f"등록되지 않은 소스 타입: {source_type}")
        return loader

    @property
    def supported_types(self) -> list[str]:
        return list(self._loaders.keys())


class RAGPipeline:
    """모든 의존성을 인터페이스(추상 클래스)로 받는다. (DIP)"""

    def __init__(self, loader_registry: LoaderRegistry, chunker: TextChunker,
                 embedder: BaseEmbedder, store: BaseStore, config: RAGConfig):
        self._loaders = loader_registry
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._config = config

    def ingest(self, source_type: str, source: str) -> dict:
        try:
            text = self._loaders.get(source_type).load(source)
            if not text.strip():
                return {"status": "error", "message": "빈 콘텐츠"}

            doc_id = hashlib.md5(text.encode()).hexdigest()
            chunks = self._chunker.split(text, doc_id, source, source_type)
            embeddings = self._embedder.embed_batch([c.content for c in chunks])

            self._store.save_document(doc_id, source, text, source_type, len(chunks))
            self._store.save_chunks(chunks, embeddings)

            logger.info(f"[{source_type.upper()}] {source} → {len(chunks)}개 청크")
            return {"status": "ok", "doc_id": doc_id, "chunks": len(chunks)}
        except Exception as e:
            logger.error(f"[{source_type.upper()}] {source} 실패: {e}")
            return {"status": "error", "message": str(e)}

    def ingest_batch(self, sources: list[dict]) -> dict:
        results = {"success": 0, "fail": 0, "details": []}
        for src in sources:
            source_type = src.get("type", "")
            source = src.get("path") or src.get("url", "")
            result = self.ingest(source_type, source)
            results["success" if result["status"] == "ok" else "fail"] += 1
            results["details"].append(result)
        return results

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self._config.default_top_k
        query_embedding = self._embedder.embed(query)
        return self._store.search_similar(query_embedding, k)

    def stats(self) -> dict:
        return self._store.get_stats()
