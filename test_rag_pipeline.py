"""RAGSystem 단위 테스트."""
import os
import json
import tempfile
import pytest
import rag_pipeline as rp


@pytest.fixture(autouse=True)
def reset_globals():
    """매 테스트마다 전역 상태 초기화."""
    rp.processed_count = 0
    rp.error_log = []
    rp.db_connection = None
    yield


@pytest.fixture
def rag(tmp_path):
    """임시 DB로 RAGSystem 생성."""
    system = rp.RAGSystem.__new__(rp.RAGSystem)
    system.documents = []
    system.chunks = []
    system.embeddings = []
    system.collection_name = "assets"
    system.embed_model = "models/text-embedding-004"
    system.db_path = str(tmp_path / "test.db")
    system._init_db()
    return system


@pytest.fixture
def sample_file(tmp_path):
    """테스트용 텍스트 파일."""
    p = tmp_path / "sample.txt"
    p.write_text("A" * 1200, encoding="utf-8")
    return str(p)


@pytest.fixture
def sample_csv(tmp_path):
    """테스트용 CSV 파일."""
    p = tmp_path / "sample.csv"
    p.write_text("name,value\nalpha,100\nbeta,200\ngamma,300\n", encoding="utf-8")
    return str(p)


class TestChunking:
    """청킹 로직 테스트 — process_pdf 내부에 묻혀있어 직접 테스트가 어려움."""

    def test_pdf_creates_chunks(self, rag, sample_file, monkeypatch):
        monkeypatch.setattr("requests.post", lambda *a, **kw: type("R", (), {
            "status_code": 200,
            "json": lambda self: {"embedding": {"values": [0.1] * 768}},
        })())
        result = rag.process_pdf(sample_file)
        assert result["status"] == "ok"
        assert result["chunks"] >= 3  # 1200자 / 500 = 최소 3개

    def test_empty_file_returns_error(self, rag, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        result = rag.process_pdf(str(p))
        assert result["status"] == "error"

    def test_csv_creates_chunks(self, rag, sample_csv, monkeypatch):
        monkeypatch.setattr("requests.post", lambda *a, **kw: type("R", (), {
            "status_code": 200,
            "json": lambda self: {"embedding": {"values": [0.1] * 768}},
        })())
        result = rag.process_csv(sample_csv)
        assert result["status"] == "ok"
        assert result["chunks"] >= 1


class TestStats:
    """통계 테스트."""

    def test_empty_stats(self, rag):
        stats = rag.get_stats()
        assert stats["total_documents"] == 0
        assert stats["total_chunks"] == 0

    def test_stats_after_ingest(self, rag, sample_file, monkeypatch):
        monkeypatch.setattr("requests.post", lambda *a, **kw: type("R", (), {
            "status_code": 200,
            "json": lambda self: {"embedding": {"values": [0.1] * 768}},
        })())
        rag.process_pdf(sample_file)
        stats = rag.get_stats()
        assert stats["total_documents"] == 1
        assert stats["total_chunks"] >= 1
        assert stats["by_type"]["pdf"] == 1


class TestProcessAll:
    """일괄 처리 테스트."""

    def test_unsupported_type(self, rag):
        result = rag.process_all([{"type": "xml", "path": "test.xml"}])
        assert result["fail"] == 1

    def test_missing_file(self, rag):
        result = rag.process_all([{"type": "pdf", "path": "/nonexistent.txt"}])
        assert result["fail"] == 1
