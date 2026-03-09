"""Phase 2 단위 테스트 — DI · Strategy · Repository 패턴 검증.

표준 라이브러리 unittest만 사용한다. (외부 테스트 프레임워크 없음)
Embedder는 MagicMock으로 대체하여 외부 API 호출 없이 테스트한다.
"""
import os
import unittest
import tempfile
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch
from config import RAGConfig, ChunkConfig, EmbedConfig, StoreConfig
from loaders import BaseLoader, PdfLoader, WebLoader, CsvLoader
from processing import TextChunker, BaseEmbedder
from processing.chunker import Chunk
from stores import BaseStore, SqliteVectorStore
from pipeline import LoaderRegistry, RAGPipeline


# ── Config ────────────────────────────────────────────────────

class TestConfig(unittest.TestCase):
    """Nested Config — 설정값 관리 및 불변성(frozen) 검증."""

    def test_nested_defaults_match_original(self):
        """기본값이 원본 매직 넘버(500, 50, 768)와 일치하는지 확인."""
        cfg = RAGConfig()
        self.assertEqual(cfg.chunk.size, 500)
        self.assertEqual(cfg.chunk.overlap, 50)
        self.assertEqual(cfg.embed.dimension, 768)
        self.assertEqual(cfg.store.db_path, "/tmp/rag_assets.db")
        self.assertEqual(cfg.default_top_k, 5)

    def test_custom_chunk_overrides_defaults(self):
        """ChunkConfig를 커스텀하면 RAGConfig에 정확히 반영되는지 확인."""
        cfg = RAGConfig(chunk=ChunkConfig(size=1000, overlap=100))
        self.assertEqual(cfg.chunk.size, 1000)
        self.assertEqual(cfg.chunk.overlap, 100)

    def test_frozen_config_prevents_mutation(self):
        """frozen=True 설정으로 런타임 값 변경이 차단되는지 확인."""
        cfg = ChunkConfig()
        with self.assertRaises(FrozenInstanceError):
            cfg.size = 999


# ── Loaders (Strategy 패턴) ──────────────────────────────────

class TestPdfLoader(unittest.TestCase):
    """PdfLoader — BaseLoader 인터페이스 구현 및 파일 읽기 검증."""

    def test_implements_base_loader_interface(self):
        """PdfLoader가 BaseLoader ABC를 구현하는지 확인. (LSP)"""
        self.assertIsInstance(PdfLoader(), BaseLoader)

    def test_source_type_returns_pdf(self):
        """source_type 프로퍼티가 'pdf' 문자열을 반환하는지 확인."""
        self.assertEqual(PdfLoader().source_type, "pdf")

    def test_load_reads_file_content_exactly(self):
        """파일 경로를 받아 텍스트 내용을 그대로 반환하는지 확인."""
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("hello PDF world")
            path = f.name
        try:
            self.assertEqual(PdfLoader().load(path), "hello PDF world")
        finally:
            os.unlink(path)


class TestCsvLoader(unittest.TestCase):
    """CsvLoader — CSV 파싱 및 텍스트 변환 검증."""

    def test_source_type_returns_csv(self):
        """source_type 프로퍼티가 'csv' 문자열을 반환하는지 확인."""
        self.assertEqual(CsvLoader().source_type, "csv")

    def test_load_converts_rows_to_key_value_text(self):
        """CSV 행이 'key: value | key: value' 형식 텍스트로 변환되는지 확인."""
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("name,score\nalice,95\nbob,87\n")
            path = f.name
        try:
            text = CsvLoader().load(path)
            self.assertIn("alice", text)
            self.assertIn("bob", text)
            self.assertIn("score: 95", text)
        finally:
            os.unlink(path)

    def test_empty_csv_with_header_only_returns_empty(self):
        """헤더만 있는 CSV 파일은 빈 문자열을 반환하는지 확인."""
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("name,score\n")
            path = f.name
        try:
            self.assertEqual(CsvLoader().load(path), "")
        finally:
            os.unlink(path)


class TestWebLoader(unittest.TestCase):
    """WebLoader — HTML 스트리핑 및 웹 크롤링 검증."""

    def test_source_type_returns_web(self):
        """source_type 프로퍼티가 'web' 문자열을 반환하는지 확인."""
        self.assertEqual(WebLoader().source_type, "web")

    @patch("loaders.web_loader.requests.get")
    def test_load_strips_script_tags_and_html(self, mock_get):
        """<script> 태그 내용이 제거되고 순수 텍스트만 남는지 확인."""
        mock_resp = MagicMock()
        mock_resp.text = (
            "<html><script>evil()</script><p>clean text</p></html>"
        )
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        text = WebLoader().load("https://example.com")
        self.assertIn("clean text", text)
        self.assertNotIn("evil", text)


# ── LoaderRegistry (OCP) ─────────────────────────────────────

class TestLoaderRegistry(unittest.TestCase):
    """LoaderRegistry — 런타임 로더 등록/조회 및 OCP 준수 검증."""

    def test_register_and_get_returns_correct_loader(self):
        """로더 등록 후 source_type으로 정확한 로더를 조회하는지 확인."""
        registry = LoaderRegistry()
        registry.register(PdfLoader())
        self.assertIsInstance(registry.get("pdf"), PdfLoader)

    def test_unregistered_type_raises_value_error(self):
        """등록되지 않은 타입 조회 시 ValueError가 발생하는지 확인."""
        registry = LoaderRegistry()
        with self.assertRaises(ValueError):
            registry.get("xml")

    def test_supported_types_lists_all_registered(self):
        """등록된 모든 타입이 supported_types 목록에 표시되는지 확인."""
        registry = LoaderRegistry()
        registry.register(PdfLoader())
        registry.register(CsvLoader())
        self.assertEqual(sorted(registry.supported_types), ["csv", "pdf"])

    def test_custom_loader_extends_without_code_change(self):
        """새 로더(JsonLoader)를 추가해도 기존 코드 수정이 불필요한지 확인. (OCP)"""

        class JsonLoader(BaseLoader):
            @property
            def source_type(self) -> str:
                return "json"

            def load(self, source: str) -> str:
                return '{"test": true}'

        registry = LoaderRegistry()
        registry.register(JsonLoader())
        self.assertEqual(registry.get("json").load("any"), '{"test": true}')


# ── TextChunker ───────────────────────────────────────────────

class TestTextChunker(unittest.TestCase):
    """TextChunker — 텍스트 분할, 오버랩, Chunk 불변성 검증."""

    def test_long_text_produces_multiple_chunks(self):
        """25자 텍스트를 10자 청크로 분할하면 3개 이상 생성되는지 확인."""
        chunker = TextChunker(ChunkConfig(size=10, overlap=2))
        chunks = chunker.split("A" * 25, "doc1", "test.txt", "pdf")
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(isinstance(c, Chunk) for c in chunks))

    def test_empty_text_returns_no_chunks(self):
        """빈 문자열을 분할하면 빈 리스트를 반환하는지 확인."""
        chunker = TextChunker(ChunkConfig(size=10, overlap=2))
        self.assertEqual(chunker.split("", "doc1", "test.txt", "pdf"), [])

    def test_adjacent_chunks_share_overlap_content(self):
        """인접 청크의 끝 3자와 다음 청크의 시작 3자가 동일한지 확인."""
        chunker = TextChunker(ChunkConfig(size=10, overlap=3))
        chunks = chunker.split("0123456789ABCDEFGHIJ", "d1", "s.txt", "pdf")
        if len(chunks) >= 2:
            self.assertEqual(chunks[0].content[-3:], chunks[1].content[:3])

    def test_chunk_dataclass_is_immutable(self):
        """Chunk가 frozen dataclass라서 속성 변경이 차단되는지 확인."""
        chunk = Chunk("text", 0, "d1", "src", "pdf")
        with self.assertRaises(FrozenInstanceError):
            chunk.content = "changed"

    def test_chunk_preserves_all_metadata(self):
        """생성된 Chunk에 doc_id, source, position이 정확히 보존되는지 확인."""
        chunker = TextChunker(ChunkConfig(size=100, overlap=0))
        chunks = chunker.split("content", "d1", "src.txt", "pdf")
        self.assertEqual(chunks[0].doc_id, "d1")
        self.assertEqual(chunks[0].source, "src.txt")
        self.assertEqual(chunks[0].position, 0)


# ── SqliteVectorStore (Repository 패턴) ──────────────────────

class TestVectorStore(unittest.TestCase):
    """SqliteVectorStore — Repository 인터페이스 구현 및 유사도 검색 검증."""

    def test_implements_base_store_interface(self):
        """SqliteVectorStore가 BaseStore ABC를 구현하는지 확인. (LSP)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SqliteVectorStore(
                StoreConfig(db_path=os.path.join(tmpdir, "t.db")))
            self.assertIsInstance(store, BaseStore)

    def test_cosine_similarity_identical_vectors(self):
        """동일한 벡터의 코사인 유사도가 1.0인지 확인."""
        score = SqliteVectorStore._cosine_similarity([1, 0, 0], [1, 0, 0])
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_cosine_similarity_orthogonal_vectors(self):
        """직교 벡터(90도)의 코사인 유사도가 0.0인지 확인."""
        score = SqliteVectorStore._cosine_similarity([1, 0], [0, 1])
        self.assertAlmostEqual(score, 0.0, places=3)

    def test_cosine_similarity_zero_vector_safe(self):
        """영벡터가 포함되면 ZeroDivision 없이 0.0을 반환하는지 확인."""
        self.assertEqual(
            SqliteVectorStore._cosine_similarity([0, 0], [1, 1]), 0.0)

    def test_save_document_reflected_in_stats(self):
        """문서 저장 후 get_stats()에 문서 수와 타입이 반영되는지 확인."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SqliteVectorStore(
                StoreConfig(db_path=os.path.join(tmpdir, "t.db")))
            store.save_document("d1", "src.txt", "content", "pdf", 2)
            stats = store.get_stats()
            self.assertEqual(stats["total_documents"], 1)
            self.assertEqual(stats["by_type"]["pdf"], 1)

    def test_save_chunks_and_search_returns_match(self):
        """청크 저장 후 동일 벡터로 검색하면 score > 0.99로 조회되는지 확인."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SqliteVectorStore(
                StoreConfig(db_path=os.path.join(tmpdir, "t.db")))
            chunks = [Chunk("hello world", 0, "d1", "src.txt", "pdf")]
            store.save_chunks(chunks, [[1.0, 0.0, 0.0]])

            results = store.search_similar([1.0, 0.0, 0.0], top_k=5)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["content"], "hello world")
            self.assertGreater(results[0]["score"], 0.99)


# ── RAGPipeline (DI 통합) ────────────────────────────────────

class TestPipeline(unittest.TestCase):
    """RAGPipeline — DI로 조립된 파이프라인 통합 검증.

    Embedder를 MagicMock으로 대체하여 외부 API 호출 없이 테스트한다.
    이는 DIP(의존성 역전 원칙)이 올바르게 적용된 증거이기도 하다.
    — 구체 클래스가 아닌 추상 인터페이스에 의존하므로 Mock 교체가 가능.
    """

    def _build_pipeline(self, tmpdir):
        """테스트용 파이프라인을 조립한다. (Embedder만 Mock으로 대체)"""
        mock_embedder = MagicMock(spec=BaseEmbedder)
        mock_embedder.embed_batch.return_value = [[0.1, 0.2, 0.3]]
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

        cfg = RAGConfig(
            store=StoreConfig(db_path=os.path.join(tmpdir, "t.db")))
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

    def test_ingest_returns_ok_and_calls_embedder(self):
        """유효한 문서 수집 시 status='ok'이고 embed_batch가 호출되는지 확인."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False,
                    encoding="utf-8") as f:
                f.write("Sample document content for testing.")
                path = f.name
            try:
                pipeline, mock_emb = self._build_pipeline(tmpdir)
                result = pipeline.ingest("pdf", path)
                self.assertEqual(result["status"], "ok")
                self.assertGreaterEqual(result["chunks"], 1)
                mock_emb.embed_batch.assert_called_once()
            finally:
                os.unlink(path)

    def test_ingest_empty_returns_error_message(self):
        """공백만 있는 문서 수집 시 '빈 콘텐츠' 에러 메시지를 반환하는지 확인."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False,
                    encoding="utf-8") as f:
                f.write("   ")
                path = f.name
            try:
                pipeline, _ = self._build_pipeline(tmpdir)
                result = pipeline.ingest("pdf", path)
                self.assertEqual(result["status"], "error")
                self.assertIn("빈 콘텐츠", result["message"])
            finally:
                os.unlink(path)

    def test_ingest_batch_counts_success_and_failure(self):
        """배치에서 유효 PDF 1건 성공 + 미등록 타입 1건 실패를 정확히 카운트하는지 확인."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False,
                    encoding="utf-8") as f:
                f.write("batch content")
                path = f.name
            try:
                pipeline, _ = self._build_pipeline(tmpdir)
                results = pipeline.ingest_batch([
                    {"type": "pdf", "path": path},
                    {"type": "xml", "path": "nope.xml"},
                ])
                self.assertEqual(results["success"], 1)
                self.assertEqual(results["fail"], 1)
            finally:
                os.unlink(path)

    def test_stats_empty_db_returns_zeros(self):
        """빈 DB에서 총 문서/청크 수가 모두 0인지 확인."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline, _ = self._build_pipeline(tmpdir)
            stats = pipeline.stats()
            self.assertEqual(stats["total_documents"], 0)
            self.assertEqual(stats["total_chunks"], 0)

    def test_search_delegates_to_embedder(self):
        """search() 호출 시 embedder.embed()에 쿼리가 위임되는지 확인. (DI 검증)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline, mock_emb = self._build_pipeline(tmpdir)
            pipeline.search("test query", top_k=3)
            mock_emb.embed.assert_called_once_with("test query")


if __name__ == "__main__":
    unittest.main()
