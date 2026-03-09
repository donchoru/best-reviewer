"""Phase 1 단위 테스트 — 각 클래스를 독립적으로 테스트."""
import pytest
from config import RAGConfig
from document_loader import DocumentLoader
from text_chunker import TextChunker, Chunk
from embedder import Embedder
from vector_store import VectorStore
from pipeline import RAGPipeline
from unittest.mock import patch, MagicMock


# ── Config ────────────────────────────────────────────────────

class TestConfig:

    def test_defaults(self):
        cfg = RAGConfig()
        assert cfg.chunk_size == 500
        assert cfg.chunk_overlap == 50
        assert cfg.embed_dimension == 768

    def test_custom(self):
        cfg = RAGConfig(chunk_size=1000, chunk_overlap=100)
        assert cfg.chunk_size == 1000


# ── DocumentLoader ────────────────────────────────────────────

class TestDocumentLoader:

    def test_load_pdf(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("hello world", encoding="utf-8")
        loader = DocumentLoader()
        assert loader.load_pdf(str(p)) == "hello world"

    def test_load_csv(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("name,val\nalpha,1\nbeta,2\n", encoding="utf-8")
        loader = DocumentLoader()
        text = loader.load_csv(str(p))
        assert "alpha" in text
        assert "beta" in text

    def test_load_unsupported(self):
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="지원하지 않는"):
            loader.load("xml", "test.xml")

    def test_load_dispatches(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("content", encoding="utf-8")
        loader = DocumentLoader()
        assert loader.load("pdf", str(p)) == "content"


# ── TextChunker ───────────────────────────────────────────────

class TestTextChunker:

    def test_split_basic(self):
        cfg = RAGConfig(chunk_size=10, chunk_overlap=2)
        chunker = TextChunker(cfg)
        chunks = chunker.split("A" * 25, "doc1", "test.txt", "pdf")
        assert len(chunks) >= 3
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_split_empty(self):
        cfg = RAGConfig(chunk_size=10, chunk_overlap=2)
        chunker = TextChunker(cfg)
        chunks = chunker.split("", "doc1", "test.txt", "pdf")
        assert len(chunks) == 0

    def test_overlap(self):
        cfg = RAGConfig(chunk_size=10, chunk_overlap=3)
        chunker = TextChunker(cfg)
        text = "0123456789ABCDEFGHIJ"  # 20자
        chunks = chunker.split(text, "doc1", "test.txt", "pdf")
        # 두 번째 청크의 시작이 첫 번째와 겹쳐야 함
        if len(chunks) >= 2:
            assert chunks[0].content[-3:] == chunks[1].content[:3]

    def test_chunk_metadata(self):
        cfg = RAGConfig(chunk_size=100, chunk_overlap=0)
        chunker = TextChunker(cfg)
        chunks = chunker.split("test content", "d1", "src.txt", "pdf")
        assert chunks[0].doc_id == "d1"
        assert chunks[0].source == "src.txt"
        assert chunks[0].doc_type == "pdf"


# ── VectorStore ───────────────────────────────────────────────

class TestVectorStore:

    def test_cosine_similarity_identical(self):
        score = VectorStore._cosine_similarity([1, 0, 0], [1, 0, 0])
        assert abs(score - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        score = VectorStore._cosine_similarity([1, 0], [0, 1])
        assert abs(score) < 0.001

    def test_cosine_similarity_zero(self):
        score = VectorStore._cosine_similarity([0, 0], [1, 1])
        assert score == 0.0

    def test_save_and_stats(self, tmp_path):
        cfg = RAGConfig(db_path=str(tmp_path / "test.db"))
        store = VectorStore(cfg)
        store.save_document("d1", "src.txt", "content", "pdf", 2)
        stats = store.get_stats()
        assert stats["total_documents"] == 1
        assert stats["by_type"]["pdf"] == 1

    def test_save_chunks_and_search(self, tmp_path):
        cfg = RAGConfig(db_path=str(tmp_path / "test.db"))
        store = VectorStore(cfg)
        chunks = [Chunk("hello world", 0, "d1", "src.txt", "pdf")]
        embeddings = [[1.0, 0.0, 0.0]]
        store.save_chunks(chunks, embeddings)

        results = store.search_similar([1.0, 0.0, 0.0], top_k=5)
        assert len(results) == 1
        assert results[0]["content"] == "hello world"
        assert results[0]["score"] > 0.99


# ── RAGPipeline (통합) ───────────────────────────────────────

class TestPipeline:

    def test_ingest_pdf(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("A" * 600, encoding="utf-8")

        cfg = RAGConfig(db_path=str(tmp_path / "test.db"))
        pipeline = RAGPipeline(cfg)

        with patch.object(pipeline.embedder, "embed_batch",
                          return_value=[[0.1] * 768, [0.2] * 768]):
            result = pipeline.ingest("pdf", str(p))

        assert result["status"] == "ok"
        assert result["chunks"] >= 2

    def test_ingest_empty(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")

        cfg = RAGConfig(db_path=str(tmp_path / "test.db"))
        pipeline = RAGPipeline(cfg)
        result = pipeline.ingest("pdf", str(p))
        assert result["status"] == "error"

    def test_ingest_batch_mixed(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("test content here", encoding="utf-8")

        cfg = RAGConfig(db_path=str(tmp_path / "test.db"))
        pipeline = RAGPipeline(cfg)

        with patch.object(pipeline.embedder, "embed_batch",
                          return_value=[[0.1] * 768]):
            results = pipeline.ingest_batch([
                {"type": "pdf", "path": str(p)},
                {"type": "xml", "path": "nope.xml"},
            ])

        assert results["success"] == 1
        assert results["fail"] == 1

    def test_stats(self, tmp_path):
        cfg = RAGConfig(db_path=str(tmp_path / "test.db"))
        pipeline = RAGPipeline(cfg)
        stats = pipeline.stats()
        assert stats["total_documents"] == 0
