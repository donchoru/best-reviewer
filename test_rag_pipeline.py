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
        with self.assertRaises(FrozenInstanceError):
            ChunkConfig().size = 999


class TestPdfLoader(unittest.TestCase):
    """PdfLoader — BaseLoader 인터페이스 구현 및 파일 읽기 검증."""
    def test_implements_base_loader_interface(self):
        """PdfLoader가 BaseLoader ABC를 구현하는지 확인. (LSP)"""
        self.assertIsInstance(PdfLoader(), BaseLoader)
    def test_source_type_returns_pdf(self):
        """source_type 프로퍼티가 'pdf'를 반환하는지 확인."""
        self.assertEqual(PdfLoader().source_type, "pdf")
    def test_load_reads_file_content_exactly(self):
        """파일 경로를 받아 텍스트를 그대로 반환하는지 확인."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("hello PDF world")
            path = f.name
        try:
            self.assertEqual(PdfLoader().load(path), "hello PDF world")
        finally:
            os.unlink(path)


class TestCsvLoader(unittest.TestCase):
    """CsvLoader — CSV 파싱 및 텍스트 변환 검증."""
    def test_source_type_returns_csv(self):
        """source_type이 'csv'를 반환하는지 확인."""
        self.assertEqual(CsvLoader().source_type, "csv")
    def test_load_converts_rows_to_key_value_text(self):
        """CSV 행이 'key: value' 형식 텍스트로 변환되는지 확인."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("name,score\nalice,95\nbob,87\n")
            path = f.name
        try:
            text = CsvLoader().load(path)
            self.assertIn("alice", text)
            self.assertIn("score: 95", text)
        finally:
            os.unlink(path)
    def test_empty_csv_with_header_only_returns_empty(self):
        """헤더만 있는 CSV는 빈 문자열을 반환하는지 확인."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("name,score\n")
            path = f.name
        try:
            self.assertEqual(CsvLoader().load(path), "")
        finally:
            os.unlink(path)


class TestWebLoader(unittest.TestCase):
    """WebLoader — HTML 스트리핑 검증."""
    def test_source_type_returns_web(self):
        """source_type이 'web'을 반환하는지 확인."""
        self.assertEqual(WebLoader().source_type, "web")
    @patch("loaders.web_loader.requests.get")
    def test_load_strips_script_tags_and_html(self, mock_get):
        """<script> 태그가 제거되고 순수 텍스트만 남는지 확인."""
        mock_resp = MagicMock()
        mock_resp.text = "<html><script>evil()</script><p>clean text</p></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        text = WebLoader().load("https://example.com")
        self.assertIn("clean text", text)
        self.assertNotIn("evil", text)


class TestLoaderRegistry(unittest.TestCase):
    """LoaderRegistry — OCP 준수 검증."""
    def test_register_and_get_returns_correct_loader(self):
        """로더 등록 후 타입으로 조회하는지 확인."""
        registry = LoaderRegistry()
        registry.register(PdfLoader())
        self.assertIsInstance(registry.get("pdf"), PdfLoader)
    def test_unregistered_type_raises_value_error(self):
        """미등록 타입 조회 시 ValueError 발생 확인."""
        with self.assertRaises(ValueError):
            LoaderRegistry().get("xml")
    def test_supported_types_lists_all_registered(self):
        """등록된 모든 타입이 목록에 표시되는지 확인."""
        registry = LoaderRegistry()
        registry.register(PdfLoader())
        registry.register(CsvLoader())
        self.assertEqual(sorted(registry.supported_types), ["csv", "pdf"])
    def test_custom_loader_extends_without_code_change(self):
        """새 로더 추가 시 기존 코드 수정 불필요 확인. (OCP)"""
        class JsonLoader(BaseLoader):
            @property
            def source_type(self) -> str: return "json"
            def load(self, source: str) -> str: return '{"test": true}'
        registry = LoaderRegistry()
        registry.register(JsonLoader())
        self.assertEqual(registry.get("json").load("any"), '{"test": true}')


class TestTextChunker(unittest.TestCase):
    """TextChunker — 텍스트 분할 및 Chunk 불변성 검증."""
    def test_long_text_produces_multiple_chunks(self):
        """25자를 10자 청크로 분할하면 3개 이상 생성되는지 확인."""
        chunks = TextChunker(ChunkConfig(size=10, overlap=2)).split("A" * 25, "d1", "t.txt", "pdf")
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(isinstance(c, Chunk) for c in chunks))
    def test_empty_text_returns_no_chunks(self):
        """빈 문자열 → 빈 리스트 확인."""
        self.assertEqual(TextChunker(ChunkConfig(size=10, overlap=2)).split("", "d1", "t.txt", "pdf"), [])
    def test_adjacent_chunks_share_overlap_content(self):
        """인접 청크 끝 3자 = 다음 청크 시작 3자 확인."""
        chunks = TextChunker(ChunkConfig(size=10, overlap=3)).split("0123456789ABCDEFGHIJ", "d1", "s.txt", "pdf")
        if len(chunks) >= 2:
            self.assertEqual(chunks[0].content[-3:], chunks[1].content[:3])
    def test_chunk_dataclass_is_immutable(self):
        """Chunk frozen → 속성 변경 차단 확인."""
        with self.assertRaises(FrozenInstanceError):
            Chunk("text", 0, "d1", "src", "pdf").content = "changed"
    def test_chunk_preserves_all_metadata(self):
        """Chunk에 doc_id, source, position 보존 확인."""
        chunks = TextChunker(ChunkConfig(size=100, overlap=0)).split("content", "d1", "src.txt", "pdf")
        self.assertEqual(chunks[0].doc_id, "d1")
        self.assertEqual(chunks[0].source, "src.txt")
        self.assertEqual(chunks[0].position, 0)


class TestVectorStore(unittest.TestCase):
    """SqliteVectorStore — Repository 패턴 및 유사도 검색 검증."""
    def test_implements_base_store_interface(self):
        """SqliteVectorStore가 BaseStore ABC를 구현하는지 확인. (LSP)"""
        with tempfile.TemporaryDirectory() as d:
            self.assertIsInstance(SqliteVectorStore(StoreConfig(db_path=os.path.join(d, "t.db"))), BaseStore)
    def test_cosine_similarity_identical_vectors(self):
        """동일 벡터 → 유사도 1.0 확인."""
        self.assertAlmostEqual(SqliteVectorStore._cosine_similarity([1, 0, 0], [1, 0, 0]), 1.0, places=3)
    def test_cosine_similarity_orthogonal_vectors(self):
        """직교 벡터 → 유사도 0.0 확인."""
        self.assertAlmostEqual(SqliteVectorStore._cosine_similarity([1, 0], [0, 1]), 0.0, places=3)
    def test_cosine_similarity_zero_vector_safe(self):
        """영벡터 → ZeroDivision 없이 0.0 반환 확인."""
        self.assertEqual(SqliteVectorStore._cosine_similarity([0, 0], [1, 1]), 0.0)
    def test_save_document_reflected_in_stats(self):
        """문서 저장 후 stats에 반영 확인."""
        with tempfile.TemporaryDirectory() as d:
            store = SqliteVectorStore(StoreConfig(db_path=os.path.join(d, "t.db")))
            store.save_document("d1", "src.txt", "content", "pdf", 2)
            stats = store.get_stats()
            self.assertEqual(stats["total_documents"], 1)
            self.assertEqual(stats["by_type"]["pdf"], 1)
    def test_save_chunks_and_search_returns_match(self):
        """청크 저장 후 동일 벡터 검색 → score > 0.99 확인."""
        with tempfile.TemporaryDirectory() as d:
            store = SqliteVectorStore(StoreConfig(db_path=os.path.join(d, "t.db")))
            store.save_chunks([Chunk("hello world", 0, "d1", "src.txt", "pdf")], [[1.0, 0.0, 0.0]])
            results = store.search_similar([1.0, 0.0, 0.0], top_k=5)
            self.assertEqual(len(results), 1)
            self.assertGreater(results[0]["score"], 0.99)


class TestPipeline(unittest.TestCase):
    """RAGPipeline — DI 통합 검증. Mock Embedder로 외부 API 없이 테스트."""
    def _build_pipeline(self, tmpdir):
        mock_embedder = MagicMock(spec=BaseEmbedder)
        mock_embedder.embed_batch.return_value = [[0.1, 0.2, 0.3]]
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
        cfg = RAGConfig(store=StoreConfig(db_path=os.path.join(tmpdir, "t.db")))
        registry = LoaderRegistry()
        registry.register(PdfLoader())
        registry.register(CsvLoader())
        pipeline = RAGPipeline(loader_registry=registry, chunker=TextChunker(cfg.chunk),
                               embedder=mock_embedder, store=SqliteVectorStore(cfg.store), config=cfg)
        return pipeline, mock_embedder
    def test_ingest_returns_ok_and_calls_embedder(self):
        """유효 문서 수집 → status='ok' + embed_batch 호출 확인."""
        with tempfile.TemporaryDirectory() as d:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
                f.write("Sample document content for testing.")
                path = f.name
            try:
                pipeline, mock_emb = self._build_pipeline(d)
                result = pipeline.ingest("pdf", path)
                self.assertEqual(result["status"], "ok")
                mock_emb.embed_batch.assert_called_once()
            finally:
                os.unlink(path)
    def test_ingest_empty_returns_error_message(self):
        """공백 문서 → '빈 콘텐츠' 에러 확인."""
        with tempfile.TemporaryDirectory() as d:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
                f.write("   ")
                path = f.name
            try:
                pipeline, _ = self._build_pipeline(d)
                result = pipeline.ingest("pdf", path)
                self.assertEqual(result["status"], "error")
                self.assertIn("빈 콘텐츠", result["message"])
            finally:
                os.unlink(path)
    def test_ingest_batch_counts_success_and_failure(self):
        """배치: 유효 PDF 1건 성공 + 미등록 타입 1건 실패 카운트 확인."""
        with tempfile.TemporaryDirectory() as d:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
                f.write("batch content")
                path = f.name
            try:
                pipeline, _ = self._build_pipeline(d)
                results = pipeline.ingest_batch([{"type": "pdf", "path": path}, {"type": "xml", "path": "nope.xml"}])
                self.assertEqual(results["success"], 1)
                self.assertEqual(results["fail"], 1)
            finally:
                os.unlink(path)
    def test_stats_empty_db_returns_zeros(self):
        """빈 DB → 문서/청크 수 0 확인."""
        with tempfile.TemporaryDirectory() as d:
            pipeline, _ = self._build_pipeline(d)
            stats = pipeline.stats()
            self.assertEqual(stats["total_documents"], 0)
    def test_search_delegates_to_embedder(self):
        """search() → embedder.embed() 위임 확인. (DI 검증)"""
        with tempfile.TemporaryDirectory() as d:
            pipeline, mock_emb = self._build_pipeline(d)
            pipeline.search("test query", top_k=3)
            mock_emb.embed.assert_called_once_with("test query")


if __name__ == "__main__":
    unittest.main()
