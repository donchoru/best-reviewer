"""진입점 — 파이프라인 실행."""
import logging
from config import RAGConfig
from pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main():
    config = RAGConfig()
    pipeline = RAGPipeline(config)

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
