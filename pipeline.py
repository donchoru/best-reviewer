"""RAG 파이프라인 — 오케스트레이션 전담."""
import hashlib
import logging
from config import RAGConfig
from document_loader import DocumentLoader
from text_chunker import TextChunker
from embedder import Embedder
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """RAG 파이프라인 오케스트레이터.

    각 단계(로딩→청킹→임베딩→저장)를 조율하며,
    자체적으로 데이터를 처리하지 않는다.
    """

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self.loader = DocumentLoader()
        self.chunker = TextChunker(self.config)
        self.embedder = Embedder(self.config)
        self.store = VectorStore(self.config)

    def ingest(self, source_type: str, source: str) -> dict:
        try:
            text = self.loader.load(source_type, source)
            if not text.strip():
                return {"status": "error", "message": "빈 콘텐츠"}

            doc_id = hashlib.md5(text.encode()).hexdigest()
            chunks = self.chunker.split(text, doc_id, source, source_type)
            embeddings = self.embedder.embed_batch([c.content for c in chunks])

            self.store.save_document(doc_id, source, text, source_type, len(chunks))
            self.store.save_chunks(chunks, embeddings)

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
            if result["status"] == "ok":
                results["success"] += 1
            else:
                results["fail"] += 1
            results["details"].append(result)
        return results

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.config.default_top_k
        query_embedding = self.embedder.embed(query)
        return self.store.search_similar(query_embedding, k)

    def stats(self) -> dict:
        return self.store.get_stats()
