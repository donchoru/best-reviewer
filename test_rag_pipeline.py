"""Phase 2 단위 테스트 — DI·Strategy·Repository 패턴 검증."""
import pytest
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch
from config import RAGConfig, ChunkConfig, EmbedConfig, StoreConfig
from loaders import BaseLoader, PdfLoader, WebLoader, CsvLoader
from processing import TextChunker, BaseEmbedder
from processing.chunker import Chunk
from stores import BaseStore, SqliteVectorStore
from pipeline import LoaderRegistry, RAGPipeline


# ── Config ────────────────────────────────────────────────────

class TestConfig:

    def test_nested_defaults(self):
        cfg = RAGConfig()
        assert cfg.chunk.size == 500
        assert cfg.chunk.overlap == 50
        assert cfg.embed.dimension == 768
        assert cfg.store.db_path == "/tmp/rag_assets.db"
        assert cfg.default_top_k == 5

    def test_custom_chunk(self):
        cfg = RAGConfig(chunk=ChunkConfig(size=1000, overlap=100))
        assert cfg.chunk.size == 1000
        assert cfg.chunk.overlap == 100

    def test_frozen(self):
        cfg = ChunkConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.size = 999


# ── Loaders (Strategy) ───────────────────────────────────────

class TestPdfLoader:

    def test_implements_base(self):
        assert isinstance(PdfLoader(), BaseLoader)

    def test_source_type(self):
        assert PdfLoader().source_type == "pdf"

    def test_load(self, tmp_path):
        p = tmp_path / "sample.txt"
        p.write_text("hello PDF world", encoding="utf-8")
        text = PdfLoader().load(str(p))
        assert text == "hello PDF world"


class TestCsvLoader:

    def test_source_type(self):
        assert CsvLoader().source_type == "csv"

    def test_load(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("name,score\nalice,95\nbob,87\n", encoding="utf-8")
        text = CsvLoader().load(str(p))
        assert "alice" in text
        assert "bob" in text
        assert "score: 95" in text

    def test_empty_csv(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("name,score\n", encoding="utf-8")
        text = CsvLoader().load(str(p))
        assert text == ""


class TestWebLoader:

    def test_source_type(self):
        assert WebLoader().source_type == "web"

    @patch("loaders.web_loader.requests.get")
    def test_load_strips_html(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.text = "<html><script>evil()</script><p>clean text</p></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        text = WebLoader().load("https://example.com")
        assert "clean text" in text
        assert "evil" not in text


# ── LoaderRegistry (OCP) ─────────────────────────────────────

class TestLoaderRegistry:

    def test_register_and_get(self):
        registry = LoaderRegistry()
        registry.register(PdfLoader())
        loader = registry.get("pdf")
        assert isinstance(loader, PdfLoader)

    def test_unregistered_raises(self):
        registry = LoaderRegistry()
        with pytest.raises(ValueError, match="등록되지 않은"):
            registry.get("xml")

    def test_supported_types(self):
        registry = LoaderRegistry()
        registry.register(PdfLoader())
        registry.register(CsvLoader())
        assert sorted(registry.supported_types) == ["csv", "pdf"]

    def test_custom_loader_ocp(self):
        """새 로더를 추가해도 기존 코드 수정 불필요 — OCP 확인."""
        class JsonLoader(BaseLoader):
            @property
            def source_type(self) -> str:
                return "json"
            def load(self, source: str) -> str:
                return '{"test": true}'

        registry = LoaderRegistry()
        registry.register(JsonLoader())
        assert registry.get("json").load("any") == '{"test": true}'


# ── TextChunker ───────────────────────────────────────────────

class TestTextChunker:

    def test_split_basic(self):
        chunker = TextChunker(ChunkConfig(size=10, overlap=2))
        chunks = chunker.split("A" * 25, "doc1", "test.txt", "pdf")
        assert len(chunks) >= 3
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_split_empty(self):
        chunker = TextChunker(ChunkConfig(size=10, overlap=2))
        assert chunker.split("", "doc1", "test.txt", "pdf") == []

    def test_overlap_content(self):
        chunker = TextChunker(ChunkConfig(size=10, overlap=3))
        chunks = chunker.split("0123456789ABCDEFGHIJ", "d1", "s.txt", "pdf")
        if len(chunks) >= 2:
            assert chunks[0].content[-3:] == chunks[1].content[:3]

    def test_chunk_is_frozen(self):
        chunk = Chunk("text", 0, "d1", "src", "pdf")
        with pytest.raises(FrozenInstanceError):
            chunk.content = "changed"

    def test_chunk_metadata(self):
        chunker = TextChunker(ChunkConfig(size=100, overlap=0))
        chunks = chunker.split("content", "d1", "src.txt", "pdf")
        assert chunks[0].doc_id == "d1"
        assert chunks[0].source == "src.txt"
        assert chunks[0].position == 0


# ── SqliteVectorStore (Repository) ────────────────────────────

class TestVectorStore:

    def test_implements_base(self, tmp_path):
        store = SqliteVectorStore(StoreConfig(db_path=str(tmp_path / "t.db")))
        assert isinstance(store, BaseStore)

    def test_cosine_identical(self):
        score = SqliteVectorStore._cosine_similarity([1, 0, 0], [1, 0, 0])
        assert abs(score - 1.0) < 0.001

    def test_cosine_orthogonal(self):
        score = SqliteVectorStore._cosine_similarity([1, 0], [0, 1])
        assert abs(score) < 0.001

    def test_cosine_zero_vector(self):
        assert SqliteVectorStore._cosine_similarity([0, 0], [1, 1]) == 0.0

    def test_save_document_and_stats(self, tmp_path):
        store = SqliteVectorStore(StoreConfig(db_path=str(tmp_path / "t.db")))
        store.save_document("d1", "src.txt", "content", "pdf", 2)
        stats = store.get_stats()
        assert stats["total_documents"] == 1
        assert stats["by_type"]["pdf"] == 1

    def test_save_chunks_and_search(self, tmp_path):
        store = SqliteVectorStore(StoreConfig(db_path=str(tmp_path / "t.db")))
        chunks = [Chunk("hello world", 0, "d1", "src.txt", "pdf")]
        embeddings = [[1.0, 0.0, 0.0]]
        store.save_chunks(chunks, embeddings)

        results = store.search_similar([1.0, 0.0, 0.0], top_k=5)
        assert len(results) == 1
        assert results[0]["content"] == "hello world"
        assert results[0]["score"] > 0.99


# ── RAGPipeline (DI 통합) ────────────────────────────────────

class TestPipeline:
    """모든 의존성을 Mock으로 주입 — DI 검증."""

    def _build_pipeline(self, tmp_path):
        mock_embedder = MagicMock(spec=BaseEmbedder)
        mock_embedder.embed_batch.return_value = [[0.1, 0.2, 0.3]]
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

        cfg = RAGConfig(store=StoreConfig(db_path=str(tmp_path / "t.db")))
        registry = LoaderRegistry()
        registry.register(PdfLoader())
        registry.register(CsvLoader())

        pipeline = RAGPipeline(
            loader_registry=registry,
            chunker=TextChunker(cfg.chunk),
            embedder=mock_embedder,
            store=SqliteVectorStore(cfg.store),
            config=cfg,
        )
        return pipeline, mock_embedder

    def test_ingest_success(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("Sample document content for testing.", encoding="utf-8")

        pipeline, mock_emb = self._build_pipeline(tmp_path)
        result = pipeline.ingest("pdf", str(p))
        assert result["status"] == "ok"
        assert result["chunks"] >= 1
        mock_emb.embed_batch.assert_called_once()

    def test_ingest_empty(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("   ", encoding="utf-8")

        pipeline, _ = self._build_pipeline(tmp_path)
        result = pipeline.ingest("pdf", str(p))
        assert result["status"] == "error"
        assert "빈 콘텐츠" in result["message"]

    def test_ingest_batch(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("batch content", encoding="utf-8")

        pipeline, _ = self._build_pipeline(tmp_path)
        results = pipeline.ingest_batch([
            {"type": "pdf", "path": str(p)},
            {"type": "xml", "path": "nope.xml"},  # 미등록 타입
        ])
        assert results["success"] == 1
        assert results["fail"] == 1

    def test_stats_empty(self, tmp_path):
        pipeline, _ = self._build_pipeline(tmp_path)
        stats = pipeline.stats()
        assert stats["total_documents"] == 0
        assert stats["total_chunks"] == 0

    def test_search_delegates_to_store(self, tmp_path):
        pipeline, mock_emb = self._build_pipeline(tmp_path)
        pipeline.search("test query", top_k=3)
        mock_emb.embed.assert_called_once_with("test query")
