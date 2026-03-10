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

    @patch("rag_pipeline.requests.get")
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


if __name__ == "__main__":
    unittest.main()
