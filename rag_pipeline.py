"""RAG 파이프라인 — DI 기반 오케스트레이터."""
import hashlib
import logging
from config import RAGConfig, StoreConfig
from loaders import BaseLoader, PdfLoader, WebLoader, CsvLoader
from processing import TextChunker, GeminiEmbedder
from stores import SqliteVectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
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


class RAGSystem:
    """RAG 파이프라인: 로딩 → 청킹 → 임베딩 → 저장 → 검색."""

    def __init__(self, config=None, db_path=None):
        self.config = config or RAGConfig()
        if db_path:
            self.config.store = StoreConfig(db_path=db_path)
        self._loaders = LoaderRegistry()
        self._loaders.register(PdfLoader())
        self._loaders.register(WebLoader())
        self._loaders.register(CsvLoader())
        self.chunker = TextChunker(self.config)
        self.embedder = GeminiEmbedder(self.config)
        self.store = SqliteVectorStore(self.config.store)

    # ── 하위 호환 위임 메서드 ──────────────────────────────

    def load_pdf(self, path):
        return self._loaders.get("pdf").load(path)

    def load_web(self, url):
        return self._loaders.get("web").load(url)

    def load_csv(self, path):
        return self._loaders.get("csv").load(path)

    def chunk_text(self, text, doc_id, source, doc_type):
        return self.chunker.split(text, doc_id, source, doc_type)

    def get_embedding(self, text):
        return self.embedder.embed(text)

    def get_embeddings_batch(self, texts):
        return self.embedder.embed_batch(texts)

    def calc_similarity(self, a, b):
        return SqliteVectorStore._cosine_similarity(a, b)

    # ── 파이프라인 ─────────────────────────────────────────

    def ingest(self, source_type, source):
        try:
            text = self._loaders.get(source_type).load(source)

            if not text.strip():
                return {"status": "error", "message": "빈 콘텐츠"}

            doc_id = hashlib.md5(text.encode()).hexdigest()
            chunks = self.chunker.split(text, doc_id, source, source_type)
            embeddings = self.embedder.embed_batch([c.content for c in chunks])

            self.store.save_document(doc_id, source, text, source_type, len(chunks))
            self.store.save_chunks(chunks, embeddings)

            logger.info(f"[{source_type.upper()}] {source} → {len(chunks)}개 청크")
            return {"status": "ok", "doc_id": doc_id, "chunks": len(chunks)}
        except ValueError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"[{source_type.upper()}] {source} 실패: {e}")
            return {"status": "error", "message": str(e)}

    def ingest_batch(self, sources):
        results = {"success": 0, "fail": 0, "details": []}
        for src in sources:
            source_type = src.get("type", "")
            source = src.get("path") or src.get("url", "")
            result = self.ingest(source_type, source)
            results["success" if result["status"] == "ok" else "fail"] += 1
            results["details"].append(result)
        return results

    def search(self, query, top_k=None):
        k = top_k or self.config.default_top_k
        query_embedding = self.embedder.embed(query)
        return self.store.search_similar(query_embedding, k)

    def get_stats(self):
        return self.store.get_stats()


if __name__ == "__main__":
    rag = RAGSystem()
    sources = [
        {"type": "pdf", "path": "data/financial_report.txt"},
        {"type": "web", "url": "https://example.com/market-analysis"},
        {"type": "csv", "path": "data/portfolio.csv"},
    ]
    results = rag.ingest_batch(sources)
    print(f"성공: {results['success']}, 실패: {results['fail']}")
