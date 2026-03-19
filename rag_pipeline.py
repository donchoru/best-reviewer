"""하위 호환 래퍼 — pipeline.py의 RAGPipeline을 사용하세요."""
from config import RAGConfig, StoreConfig, ChunkConfig, EmbedConfig
from loaders import PdfLoader, WebLoader, CsvLoader
from processing import TextChunker, GeminiEmbedder
from stores import SqliteVectorStore
from pipeline import RAGPipeline, LoaderRegistry


class RAGSystem:
    """레거시 인터페이스 — 신규 코드는 main.create_pipeline()을 사용하세요."""

    def __init__(self, config=None, db_path=None):
        self.config = config or RAGConfig()
        if db_path:
            self.config = RAGConfig(store=StoreConfig(db_path=db_path))

        registry = LoaderRegistry()
        registry.register(PdfLoader())
        registry.register(WebLoader())
        registry.register(CsvLoader())

        self._pipeline = RAGPipeline(
            loader_registry=registry,
            chunker=TextChunker(self.config.chunk),
            embedder=GeminiEmbedder(self.config.embed),
            store=SqliteVectorStore(self.config.store),
            config=self.config,
        )
        self._registry = registry

    def load_pdf(self, path):
        return self._registry.get("pdf").load(path)

    def load_web(self, url):
        return self._registry.get("web").load(url)

    def load_csv(self, path):
        return self._registry.get("csv").load(path)

    def chunk_text(self, text, doc_id, source, doc_type):
        return self._pipeline._chunker.split(text, doc_id, source, doc_type)

    def get_embedding(self, text):
        return self._pipeline._embedder.embed(text)

    def get_embeddings_batch(self, texts):
        return self._pipeline._embedder.embed_batch(texts)

    def calc_similarity(self, a, b):
        return self._pipeline._store._calc_similarity(a, b)

    def ingest(self, source_type, source):
        return self._pipeline.ingest(source_type, source)

    def ingest_batch(self, sources):
        return self._pipeline.ingest_batch(sources)

    def search(self, query, top_k=None):
        return self._pipeline.search(query, top_k)

    def get_stats(self):
        return self._pipeline.stats()
