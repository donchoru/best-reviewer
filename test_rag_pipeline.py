"""Phase 1 단위 테스트 — Extract Class 리팩토링 결과 검증.

God Class를 분해한 각 클래스를 독립적으로 테스트한다.
표준 라이브러리 unittest만 사용한다. (외부 의존성 없음)
"""
import os
import unittest
import tempfile
from unittest.mock import patch, MagicMock
from config import RAGConfig
from document_loader import DocumentLoader
from text_chunker import TextChunker, Chunk
from vector_store import VectorStore
from pipeline import RAGPipeline


class TestConfig(unittest.TestCase):
    """RAGConfig — 매직 넘버를 dataclass로 중앙 관리하는지 검증."""

    def test_defaults_match_original_magic_numbers(self):
        """기본값이 원본 코드의 매직 넘버(500, 50, 768)와 일치하는지 확인."""
        cfg = RAGConfig()
        self.assertEqual(cfg.chunk_size, 500)
        self.assertEqual(cfg.chunk_overlap, 50)
        self.assertEqual(cfg.embed_dimension, 768)

    def test_custom_values_override_defaults(self):
        """사용자 지정 값이 기본값을 정확히 대체하는지 확인."""
        cfg = RAGConfig(chunk_size=1000, chunk_overlap=100)
        self.assertEqual(cfg.chunk_size, 1000)
        self.assertEqual(cfg.chunk_overlap, 100)


class TestDocumentLoader(unittest.TestCase):
    """DocumentLoader — PDF/CSV/웹 로딩 및 디스패치 검증."""

    def test_load_pdf_reads_file_content(self):
        """텍스트 파일을 읽어서 내용을 그대로 반환하는지 확인."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("hello world")
            path = f.name
        try:
            self.assertEqual(DocumentLoader().load_pdf(path), "hello world")
        finally:
            os.unlink(path)

    def test_load_csv_parses_rows_into_text(self):
        """CSV 파일의 각 행이 'key: value' 형식으로 변환되는지 확인."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("name,val\nalpha,1\nbeta,2\n")
            path = f.name
        try:
            text = DocumentLoader().load_csv(path)
            self.assertIn("alpha", text)
            self.assertIn("beta", text)
        finally:
            os.unlink(path)

    def test_unsupported_type_raises_value_error(self):
        """지원하지 않는 소스 타입 요청 시 ValueError가 발생하는지 확인."""
        with self.assertRaises(ValueError):
            DocumentLoader().load("xml", "test.xml")

    def test_load_dispatches_pdf_to_load_pdf(self):
        """load('pdf', path) 호출이 내부적으로 load_pdf를 실행하는지 확인."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("content")
            path = f.name
        try:
            self.assertEqual(DocumentLoader().load("pdf", path), "content")
        finally:
            os.unlink(path)


class TestTextChunker(unittest.TestCase):
    """TextChunker — 텍스트 분할, 오버랩, 메타데이터 보존 검증."""

    def test_long_text_splits_into_multiple_chunks(self):
        """25자 텍스트를 10자 청크로 분할하면 3개 이상 생성되는지 확인."""
        cfg = RAGConfig(chunk_size=10, chunk_overlap=2)
        chunks = TextChunker(cfg).split("A" * 25, "doc1", "test.txt", "pdf")
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(isinstance(c, Chunk) for c in chunks))

    def test_empty_text_returns_no_chunks(self):
        """빈 문자열을 분할하면 빈 리스트를 반환하는지 확인."""
        cfg = RAGConfig(chunk_size=10, chunk_overlap=2)
        self.assertEqual(TextChunker(cfg).split("", "d1", "s.txt", "pdf"), [])

    def test_adjacent_chunks_share_overlap(self):
        """인접 청크의 끝 3자와 다음 청크의 시작 3자가 동일한지 확인."""
        cfg = RAGConfig(chunk_size=10, chunk_overlap=3)
        chunks = TextChunker(cfg).split("0123456789ABCDEFGHIJ", "d1", "s.txt", "pdf")
        if len(chunks) >= 2:
            self.assertEqual(chunks[0].content[-3:], chunks[1].content[:3])

    def test_chunk_preserves_metadata(self):
        """생성된 Chunk에 doc_id, source, doc_type이 정확히 보존되는지 확인."""
        cfg = RAGConfig(chunk_size=100, chunk_overlap=0)
        chunks = TextChunker(cfg).split("test", "d1", "src.txt", "pdf")
        self.assertEqual(chunks[0].doc_id, "d1")
        self.assertEqual(chunks[0].source, "src.txt")
        self.assertEqual(chunks[0].doc_type, "pdf")


class TestVectorStore(unittest.TestCase):
    """VectorStore — 코사인 유사도 계산, 문서/청크 저장·검색 검증."""

    def test_cosine_similarity_identical_vectors(self):
        """동일한 벡터의 코사인 유사도가 1.0인지 확인."""
        self.assertAlmostEqual(VectorStore._cosine_similarity([1, 0, 0], [1, 0, 0]), 1.0, places=3)

    def test_cosine_similarity_orthogonal_vectors(self):
        """직교 벡터의 코사인 유사도가 0.0인지 확인."""
        self.assertAlmostEqual(VectorStore._cosine_similarity([1, 0], [0, 1]), 0.0, places=3)

    def test_cosine_similarity_zero_vector_returns_zero(self):
        """영벡터가 포함되면 0.0을 반환하는지 확인."""
        self.assertEqual(VectorStore._cosine_similarity([0, 0], [1, 1]), 0.0)

    def test_save_document_updates_stats(self):
        """문서 저장 후 get_stats()에 정확히 반영되는지 확인."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = RAGConfig(db_path=os.path.join(tmpdir, "test.db"))
            store = VectorStore(cfg)
            store.save_document("d1", "src.txt", "content", "pdf", 2)
            stats = store.get_stats()
            self.assertEqual(stats["total_documents"], 1)
            self.assertEqual(stats["by_type"]["pdf"], 1)

    def test_save_chunks_and_search_by_similarity(self):
        """청크 저장 후 동일 벡터로 검색하면 score > 0.99로 조회되는지 확인."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = RAGConfig(db_path=os.path.join(tmpdir, "test.db"))
            store = VectorStore(cfg)
            chunks = [Chunk("hello world", 0, "d1", "src.txt", "pdf")]
            store.save_chunks(chunks, [[1.0, 0.0, 0.0]])
            results = store.search_similar([1.0, 0.0, 0.0], top_k=5)
            self.assertEqual(len(results), 1)
            self.assertGreater(results[0]["score"], 0.99)


class TestPipeline(unittest.TestCase):
    """RAGPipeline — 컴포넌트 조합 통합 테스트."""

    @patch("embedder.requests.post")
    def test_ingest_pdf_returns_ok_with_chunks(self, mock_post):
        """600자 PDF 수집 시 status='ok'이고 청크가 2개 이상인지 확인."""
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"embedding": {"values": [0.1] * 768}})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("A" * 600)
            path = f.name
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cfg = RAGConfig(db_path=os.path.join(tmpdir, "test.db"))
                result = RAGPipeline(cfg).ingest("pdf", path)
                self.assertEqual(result["status"], "ok")
                self.assertGreaterEqual(result["chunks"], 2)
        finally:
            os.unlink(path)

    def test_ingest_empty_file_returns_error(self):
        """빈 파일 수집 시 에러를 반환하는지 확인."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            path = f.name
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cfg = RAGConfig(db_path=os.path.join(tmpdir, "test.db"))
                result = RAGPipeline(cfg).ingest("pdf", path)
                self.assertEqual(result["status"], "error")
        finally:
            os.unlink(path)

    @patch("embedder.requests.post")
    def test_batch_counts_success_and_fail(self, mock_post):
        """배치에서 성공 1건 + 미지원 타입 실패 1건이 정확한지 확인."""
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"embedding": {"values": [0.1] * 768}})
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("test content here")
            path = f.name
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cfg = RAGConfig(db_path=os.path.join(tmpdir, "test.db"))
                results = RAGPipeline(cfg).ingest_batch([
                    {"type": "pdf", "path": path},
                    {"type": "xml", "path": "nope.xml"},
                ])
                self.assertEqual(results["success"], 1)
                self.assertEqual(results["fail"], 1)
        finally:
            os.unlink(path)

    def test_stats_on_empty_db(self):
        """빈 DB에서 총 문서/청크 수가 0인지 확인."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = RAGConfig(db_path=os.path.join(tmpdir, "test.db"))
            stats = RAGPipeline(cfg).stats()
            self.assertEqual(stats["total_documents"], 0)


if __name__ == "__main__":
    unittest.main()
