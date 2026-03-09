"""RAGSystem 단위 테스트 — 코드 스멜이 있는 원본 코드를 검증한다.

표준 라이브러리 unittest만 사용한다. (외부 테스트 프레임워크 없음)
"""
import os
import unittest
import tempfile
from unittest.mock import patch, MagicMock
from rag_pipeline import RAGSystem


class TestChunking(unittest.TestCase):
    """텍스트를 500자 단위로 분할하는 청킹 로직을 검증한다.

    RAGSystem은 process_pdf/web/csv 내부에서 동일한 청킹 로직을
    3회 중복 구현하고 있다. 여기서는 process_pdf를 통해 간접 테스트한다.
    """

    def setUp(self):
        """테스트마다 DB를 초기화하여 테스트 간 격리를 보장한다."""
        if os.path.exists("/tmp/rag_assets.db"):
            os.unlink("/tmp/rag_assets.db")

    @patch("rag_pipeline.requests.post")
    def test_long_text_splits_into_multiple_chunks(self, mock_post):
        """1200자 텍스트 → 최소 3개 청크(500자 단위)로 분할되는지 확인."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"embedding": {"values": [0.1] * 768}})

        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("A" * 1200)
            path = f.name

        try:
            rag = RAGSystem()
            result = rag.process_pdf(path)
            self.assertEqual(result["status"], "ok")
            self.assertGreaterEqual(result["chunks"], 3)
        finally:
            os.unlink(path)

    @patch("rag_pipeline.requests.post")
    def test_short_text_produces_single_chunk(self, mock_post):
        """500자 미만 텍스트 → 1개 청크만 생성되는지 확인."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"embedding": {"values": [0.1] * 768}})

        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("short text")
            path = f.name

        try:
            rag = RAGSystem()
            result = rag.process_pdf(path)
            self.assertEqual(result["chunks"], 1)
        finally:
            os.unlink(path)

    def test_empty_file_returns_error(self):
        """빈 파일을 처리하면 에러 상태를 반환하는지 확인."""
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            path = f.name

        try:
            rag = RAGSystem()
            result = rag.process_pdf(path)
            self.assertEqual(result["status"], "error")
        finally:
            os.unlink(path)


class TestStats(unittest.TestCase):
    """문서 통계 조회 기능을 검증한다."""

    def setUp(self):
        if os.path.exists("/tmp/rag_assets.db"):
            os.unlink("/tmp/rag_assets.db")

    def test_initial_stats_are_zero(self):
        """문서가 없는 초기 상태에서 총 문서/청크 수가 0인지 확인."""
        rag = RAGSystem()
        stats = rag.get_stats()
        self.assertEqual(stats["total_documents"], 0)
        self.assertEqual(stats["total_chunks"], 0)

    @patch("rag_pipeline.requests.post")
    def test_stats_increase_after_processing(self, mock_post):
        """문서 처리 후 통계가 1 이상으로 증가하는지 확인."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"embedding": {"values": [0.1] * 768}})

        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("stats test content")
            path = f.name

        try:
            rag = RAGSystem()
            rag.process_pdf(path)
            stats = rag.get_stats()
            self.assertGreaterEqual(stats["total_documents"], 1)
            self.assertIn("pdf", stats["by_type"])
        finally:
            os.unlink(path)


class TestProcessAll(unittest.TestCase):
    """여러 소스를 한번에 처리하는 배치 기능을 검증한다."""

    def setUp(self):
        if os.path.exists("/tmp/rag_assets.db"):
            os.unlink("/tmp/rag_assets.db")

    @patch("rag_pipeline.requests.post")
    def test_valid_pdf_batch_succeeds(self, mock_post):
        """유효한 PDF가 포함된 배치 처리가 success=1을 반환하는지 확인."""
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"embedding": {"values": [0.1] * 768}})

        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("batch document content")
            path = f.name

        try:
            rag = RAGSystem()
            results = rag.process_all([{"type": "pdf", "path": path}])
            self.assertEqual(results["success"], 1)
            self.assertEqual(results["fail"], 0)
        finally:
            os.unlink(path)

    def test_unsupported_type_counted_as_fail(self):
        """지원하지 않는 타입(xml)이 fail로 카운트되는지 확인."""
        rag = RAGSystem()
        results = rag.process_all([{"type": "xml", "path": "test.xml"}])
        self.assertEqual(results["fail"], 1)
        self.assertEqual(results["success"], 0)


if __name__ == "__main__":
    unittest.main()
