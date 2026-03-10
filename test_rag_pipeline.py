"""RAGSystem 기본 테스트.

표준 라이브러리 unittest만 사용. Embedder는 MagicMock으로 대체하여
외부 API 호출 없이 테스트.
"""
import os
import unittest
import tempfile
from unittest.mock import patch, MagicMock
from rag_pipeline import RAGSystem


class TestRAGSystem(unittest.TestCase):

    def _make_system(self, tmpdir):
        return RAGSystem(db_path=os.path.join(tmpdir, "test.db"))

    def test_load_pdf_reads_file(self):
        """파일 경로를 받아 텍스트를 그대로 반환하는지 확인."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("hello PDF world")
            path = f.name
        try:
            with tempfile.TemporaryDirectory() as d:
                rag = self._make_system(d)
                self.assertEqual(rag.load_pdf(path), "hello PDF world")
        finally:
            os.unlink(path)

    def test_load_csv_converts_to_text(self):
        """CSV 행이 'key: value' 형식 텍스트로 변환되는지 확인."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("name,score\nalice,95\nbob,87\n")
            path = f.name
        try:
            with tempfile.TemporaryDirectory() as d:
                rag = self._make_system(d)
                text = rag.load_csv(path)
                self.assertIn("alice", text)
                self.assertIn("score: 95", text)
        finally:
            os.unlink(path)

    @patch("loaders.web_loader.requests.get")
    def test_load_web_strips_html(self, mock_get):
        """<script> 태그가 제거되고 순수 텍스트만 남는지 확인."""
        mock_resp = MagicMock()
        mock_resp.text = "<html><script>evil()</script><p>clean text</p></html>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp
        with tempfile.TemporaryDirectory() as d:
            rag = self._make_system(d)
            text = rag.load_web("https://example.com")
            self.assertIn("clean text", text)
            self.assertNotIn("evil", text)

    def test_chunk_text_produces_chunks(self):
        """긴 텍스트가 여러 청크로 분할되는지 확인."""
        with tempfile.TemporaryDirectory() as d:
            rag = self._make_system(d)
            chunks = rag.chunk_text("A" * 1200, "d1", "test.txt", "pdf")
            self.assertGreaterEqual(len(chunks), 3)

    def test_calc_similarity_identical(self):
        """동일 벡터 → 유사도 1.0 확인."""
        with tempfile.TemporaryDirectory() as d:
            rag = self._make_system(d)
            self.assertAlmostEqual(
                rag.calc_similarity([1, 0, 0], [1, 0, 0]), 1.0, places=3)

    def test_ingest_with_mock_embedder(self):
        """유효 문서 수집 → status='ok' 확인 (임베딩 모킹)."""
        with tempfile.TemporaryDirectory() as d:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
                f.write("Sample document content for testing.")
                path = f.name
            try:
                rag = self._make_system(d)
                rag.get_embeddings_batch = MagicMock(return_value=[[0.1, 0.2, 0.3]])
                result = rag.ingest("pdf", path)
                self.assertEqual(result["status"], "ok")
            finally:
                os.unlink(path)

    def test_get_stats_empty_db(self):
        """빈 DB → 문서/청크 수 0 확인."""
        with tempfile.TemporaryDirectory() as d:
            rag = self._make_system(d)
            stats = rag.get_stats()
            self.assertEqual(stats["total_documents"], 0)


class TestLoaders(unittest.TestCase):
    """개별 로더 단위 테스트."""

    def test_pdf_loader_reads_file_content(self):
        """PdfLoader가 파일 내용을 정확히 반환하는지 확인."""
        from loaders import PdfLoader
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("test content")
            path = f.name
        try:
            self.assertEqual(PdfLoader().load(path), "test content")
        finally:
            os.unlink(path)

    def test_csv_loader_formats_as_key_value(self):
        """CsvLoader가 key: value 형식으로 변환하는지 확인."""
        from loaders import CsvLoader
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("col1,col2\nval1,val2\n")
            path = f.name
        try:
            self.assertIn("col1: val1", CsvLoader().load(path))
        finally:
            os.unlink(path)

    def test_empty_csv_returns_empty_string(self):
        """완전히 빈 CSV 파일 → 빈 문자열 반환 확인."""
        from loaders import CsvLoader
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("")
            path = f.name
        try:
            self.assertEqual(CsvLoader().load(path), "")
        finally:
            os.unlink(path)

    def test_each_loader_has_source_type(self):
        """각 로더의 source_type 프로퍼티 확인."""
        from loaders import PdfLoader, WebLoader, CsvLoader
        self.assertEqual(PdfLoader().source_type, "pdf")
        self.assertEqual(WebLoader().source_type, "web")
        self.assertEqual(CsvLoader().source_type, "csv")


class TestTextChunker(unittest.TestCase):
    """TextChunker 단위 테스트."""

    def test_long_text_produces_multiple_chunks(self):
        """25자를 10자 청크로 분할하면 3개 이상 생성되는지 확인."""
        from config import ChunkConfig
        from processing import TextChunker
        cfg = ChunkConfig(size=10, overlap=2)
        chunks = TextChunker(cfg).split("A" * 25, "d1", "t.txt", "pdf")
        self.assertGreaterEqual(len(chunks), 3)

    def test_empty_text_returns_no_chunks(self):
        """빈 문자열 → 빈 리스트 확인."""
        from config import ChunkConfig
        from processing import TextChunker
        cfg = ChunkConfig(size=10, overlap=2)
        self.assertEqual(TextChunker(cfg).split("", "d1", "t.txt", "pdf"), [])

    def test_adjacent_chunks_share_overlap_content(self):
        """인접 청크 끝부분 = 다음 청크 시작부분 확인 (overlap)."""
        from config import ChunkConfig
        from processing import TextChunker
        cfg = ChunkConfig(size=10, overlap=3)
        chunks = TextChunker(cfg).split("0123456789ABCDEFGHIJ", "d1", "s.txt", "pdf")
        if len(chunks) >= 2:
            self.assertEqual(chunks[0].content[-3:], chunks[1].content[:3])


class TestVectorStore(unittest.TestCase):
    """SqliteVectorStore 단위 테스트."""

    def test_cosine_similarity_identical_vectors(self):
        """동일 벡터 → 유사도 1.0 확인."""
        from stores import SqliteVectorStore
        self.assertAlmostEqual(
            SqliteVectorStore._cosine_similarity([1, 0, 0], [1, 0, 0]), 1.0, places=3)

    def test_cosine_similarity_orthogonal_vectors(self):
        """직교 벡터 → 유사도 0.0 확인."""
        from stores import SqliteVectorStore
        self.assertAlmostEqual(
            SqliteVectorStore._cosine_similarity([1, 0], [0, 1]), 0.0, places=3)

    def test_cosine_similarity_zero_vector_safe(self):
        """영벡터 → ZeroDivision 없이 0.0 반환 확인."""
        from stores import SqliteVectorStore
        self.assertEqual(SqliteVectorStore._cosine_similarity([0, 0], [1, 1]), 0.0)

    def test_save_document_reflected_in_stats(self):
        """문서 저장 후 stats에 반영 확인."""
        from stores import SqliteVectorStore
        with tempfile.TemporaryDirectory() as d:
            store = SqliteVectorStore(os.path.join(d, "t.db"))
            store.save_document("d1", "src.txt", "content", "pdf", 2)
            stats = store.get_stats()
            self.assertEqual(stats["total_documents"], 1)
            self.assertEqual(stats["by_type"]["pdf"], 1)

    def test_save_chunks_and_search_returns_match(self):
        """청크 저장 후 동일 벡터 검색 → score > 0.99 확인."""
        from stores import SqliteVectorStore
        from processing.chunker import Chunk
        with tempfile.TemporaryDirectory() as d:
            store = SqliteVectorStore(os.path.join(d, "t.db"))
            store.save_chunks([Chunk("hello world", 0, "d1", "src.txt", "pdf")],
                              [[1.0, 0.0, 0.0]])
            results = store.search_similar([1.0, 0.0, 0.0], top_k=5)
            self.assertEqual(len(results), 1)
            self.assertGreater(results[0]["score"], 0.99)


if __name__ == "__main__":
    unittest.main()
