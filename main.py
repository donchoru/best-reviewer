"""진입점."""
import logging
from config import RAGConfig
from pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main():
    pipeline = RAGPipeline(RAGConfig())

    sources = [
        {"type": "pdf", "path": "data/financial_report.txt"},
        {"type": "web", "url": "https://example.com/market-analysis"},
        {"type": "csv", "path": "data/portfolio.csv"},
    ]

    results = pipeline.ingest_batch(sources)
    print(f"성공: {results['success']}, 실패: {results['fail']}")
    print(pipeline.stats())


if __name__ == "__main__":
    main()
