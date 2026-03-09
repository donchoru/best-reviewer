"""진입점 — Composition Root (DI 조립)."""
import logging
from config import RAGConfig
from loaders import PdfLoader, WebLoader, CsvLoader
from processing import TextChunker, GeminiEmbedder
from stores import SqliteVectorStore
from pipeline import RAGPipeline, LoaderRegistry

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def create_pipeline(config: RAGConfig | None = None) -> RAGPipeline:
    """파이프라인 팩토리 — 의존성 조립."""
    cfg = config or RAGConfig()

    registry = LoaderRegistry()
    registry.register(PdfLoader())
    registry.register(WebLoader())
    registry.register(CsvLoader())

    return RAGPipeline(
        loader_registry=registry,
        chunker=TextChunker(cfg.chunk),
        embedder=GeminiEmbedder(cfg.embed),
        store=SqliteVectorStore(cfg.store),
        config=cfg,
    )


def main():
    pipeline = create_pipeline()

    sources = [
        {"type": "pdf", "path": "data/financial_report.txt"},
        {"type": "web", "url": "https://example.com/market-analysis"},
        {"type": "csv", "path": "data/portfolio.csv"},
    ]

    results = pipeline.ingest_batch(sources)
    print(f"성공: {results['success']}, 실패: {results['fail']}")
    print(pipeline.stats())

    hits = pipeline.search("비정형 자산 포트폴리오 분석")
    for h in hits:
        print(f"[{h['score']:.3f}] {h['content'][:100]}...")


if __name__ == "__main__":
    main()
