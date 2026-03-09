"""비정형 자산 수집 RAG 파이프라인.

Step 2: DocumentLoader, TextChunker 추출 — 3회 중복 로딩·청킹 로직 제거.
임베딩 API 호출과 DB 저장 코드는 아직 인라인으로 남아있다.
"""
import os
import json
import hashlib
import sqlite3
import requests
from datetime import datetime
from config import RAGConfig
from document_loader import DocumentLoader
from text_chunker import TextChunker


processed_count = 0
error_log = []
db_connection = None


class RAGSystem:

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self.loader = DocumentLoader()
        self.chunker = TextChunker(self.config)
        self.documents = []
        self.embeddings = []
        self._init_db()

    def _init_db(self):
        global db_connection
        db_connection = sqlite3.connect(self.config.db_path)
        db_connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, source TEXT, content TEXT,
                doc_type TEXT, created_at TEXT, chunk_count INTEGER, status TEXT
            )
        """)
        db_connection.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, doc_id TEXT, content TEXT,
                embedding TEXT, position INTEGER
            )
        """)
        db_connection.commit()

    def _embed_chunk(self, text: str) -> list[float]:
        """단일 텍스트의 임베딩 벡터를 반환한다."""
        api_key = os.environ.get(self.config.api_key_env, "")
        url = f"{self.config.embed_api_url}/{self.config.embed_model}:embedContent?key={api_key}"
        try:
            resp = requests.post(url, json={
                "model": self.config.embed_model,
                "content": {"parts": [{"text": text}]}
            }, timeout=self.config.embed_timeout)
            if resp.status_code == 200:
                return resp.json()["embedding"]["values"]
        except Exception as e:
            error_log.append(f"임베딩 에러: {e}")
        return [0.0] * self.config.embed_dimension

    def _ingest(self, source_type: str, source: str) -> dict:
        """로딩 → 청킹 → 임베딩 → DB 저장 공통 파이프라인."""
        global processed_count, error_log

        try:
            text = self.loader.load(source_type, source)
        except Exception as e:
            error_log.append(f"[{source_type.upper()}] 로딩 실패: {source} - {e}")
            return {"status": "error", "message": str(e)}

        if not text.strip():
            return {"status": "error", "message": "빈 콘텐츠"}

        doc_id = hashlib.md5(text.encode()).hexdigest()
        chunks = self.chunker.split(text, doc_id, source, source_type)
        embeddings = [self._embed_chunk(c.content) for c in chunks]

        global db_connection
        db_connection.execute(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, source, text[:1000], source_type,
             datetime.now().isoformat(), len(chunks), "processed"))
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = hashlib.md5(chunk.content.encode()).hexdigest()
            db_connection.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)",
                (chunk_id, doc_id, chunk.content,
                 json.dumps(embedding), chunk.position))
        db_connection.commit()

        processed_count += 1
        return {"status": "ok", "doc_id": doc_id, "chunks": len(chunks)}

    def process_pdf(self, file_path: str) -> dict:
        return self._ingest("pdf", file_path)

    def process_web(self, url: str) -> dict:
        return self._ingest("web", url)

    def process_csv(self, file_path: str) -> dict:
        return self._ingest("csv", file_path)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.config.default_top_k
        query_embedding = self._embed_chunk(query)
        if all(v == 0.0 for v in query_embedding):
            return []

        global db_connection
        cursor = db_connection.execute(
            "SELECT id, doc_id, content, embedding FROM chunks")
        results = []
        for row in cursor:
            chunk_embedding = json.loads(row[3])
            if not chunk_embedding or all(v == 0.0 for v in chunk_embedding):
                continue
            dot = sum(a * b for a, b in zip(query_embedding, chunk_embedding))
            norm_q = sum(a * a for a in query_embedding) ** 0.5
            norm_c = sum(a * a for a in chunk_embedding) ** 0.5
            if norm_q == 0 or norm_c == 0:
                continue
            similarity = dot / (norm_q * norm_c)
            results.append({
                "chunk_id": row[0], "doc_id": row[1],
                "content": row[2], "score": similarity,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def get_stats(self) -> dict:
        global db_connection, processed_count, error_log
        doc_count = db_connection.execute(
            "SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = db_connection.execute(
            "SELECT COUNT(*) FROM chunks").fetchone()[0]
        type_counts = {}
        for row in db_connection.execute(
                "SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type"):
            type_counts[row[0]] = row[1]
        return {
            "total_documents": doc_count, "total_chunks": chunk_count,
            "by_type": type_counts, "processed_count": processed_count,
            "error_count": len(error_log), "errors": error_log[-10:],
        }

    def process_all(self, sources: list[dict]) -> dict:
        results = {"success": 0, "fail": 0, "details": []}
        for source in sources:
            source_type = source.get("type", "")
            src = source.get("path") or source.get("url", "")
            result = self._ingest(source_type, src)
            if result["status"] == "ok":
                results["success"] += 1
            else:
                results["fail"] += 1
            results["details"].append(result)
        return results


if __name__ == "__main__":
    rag = RAGSystem()
    sources = [
        {"type": "pdf", "path": "data/financial_report.txt"},
        {"type": "web", "url": "https://example.com/market-analysis"},
        {"type": "csv", "path": "data/portfolio.csv"},
    ]
    rag.process_all(sources)
    print(rag.get_stats())
