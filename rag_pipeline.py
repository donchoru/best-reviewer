"""RAG 파이프라인 — 비정형 자산 수집·임베딩·검색."""
import os
import re
import csv
import json
import hashlib
import sqlite3
import logging
import requests
from datetime import datetime
from config import RAGConfig
from document_loader import DocumentLoader
from processing import TextChunker

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG 파이프라인: 청킹 → 임베딩 → 저장 → 검색."""

    def __init__(self, config=None, db_path=None):
        self.config = config or RAGConfig()
        if db_path:
            self.config.db_path = db_path
        self.loader = DocumentLoader()
        self.chunker = TextChunker(self.config)
        self.api_key = os.environ.get(self.config.api_key_env, "")
        self.conn = sqlite3.connect(self.config.db_path)
        self._init_tables()

    def _init_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, source TEXT, content TEXT,
                doc_type TEXT, created_at TEXT, chunk_count INTEGER, status TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, doc_id TEXT, content TEXT,
                embedding TEXT, position INTEGER
            )
        """)
        self.conn.commit()

    # ── 문서 로딩 (DocumentLoader에 위임) ─────────────────

    def load_pdf(self, path):
        return self.loader.load("pdf", path)

    def load_web(self, url):
        return self.loader.load("web", url)

    def load_csv(self, path):
        return self.loader.load("csv", path)

    # ── 텍스트 청킹 (TextChunker에 위임) ─────────────────

    def chunk_text(self, text, doc_id, source, doc_type):
        return self.chunker.split(text, doc_id, source, doc_type)

    # ── 임베딩 ─────────────────────────────────────────────

    def get_embedding(self, text):
        url = f"{self.config.embed_api_url}/{self.config.embed_model}:embedContent?key={self.api_key}"
        resp = requests.post(url, json={
            "model": self.config.embed_model,
            "content": {"parts": [{"text": text}]},
        }, timeout=self.config.embed_timeout)
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]

    def get_embeddings_batch(self, texts):
        results = []
        for text in texts:
            try:
                results.append(self.get_embedding(text))
            except Exception:
                results.append([0.0] * self.config.embed_dimension)
        return results

    # ── 저장 ───────────────────────────────────────────────

    def save_document(self, doc_id, source, content, doc_type, chunk_count):
        self.conn.execute(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, source, content[:1000], doc_type,
             datetime.now().isoformat(), chunk_count, "processed"),
        )
        self.conn.commit()

    def save_chunks(self, chunks, embeddings):
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = hashlib.md5(chunk.content.encode()).hexdigest()
            self.conn.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)",
                (chunk_id, chunk.doc_id, chunk.content,
                 json.dumps(embedding), chunk.position),
            )
        self.conn.commit()

    # ── 검색 ───────────────────────────────────────────────

    def calc_similarity(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def search(self, query, top_k=None):
        k = top_k or self.config.default_top_k
        query_embedding = self.get_embedding(query)
        cursor = self.conn.execute(
            "SELECT id, doc_id, content, embedding FROM chunks")
        results = []
        for row in cursor:
            chunk_emb = json.loads(row[3])
            if not chunk_emb or all(v == 0.0 for v in chunk_emb):
                continue
            score = self.calc_similarity(query_embedding, chunk_emb)
            results.append({"chunk_id": row[0], "doc_id": row[1],
                            "content": row[2], "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    # ── 파이프라인 ─────────────────────────────────────────

    def ingest(self, source_type, source):
        try:
            text = self.loader.load(source_type, source)

            if not text.strip():
                return {"status": "error", "message": "빈 콘텐츠"}

            doc_id = hashlib.md5(text.encode()).hexdigest()
            chunks = self.chunk_text(text, doc_id, source, source_type)
            embeddings = self.get_embeddings_batch([c.content for c in chunks])

            self.save_document(doc_id, source, text, source_type, len(chunks))
            self.save_chunks(chunks, embeddings)

            logger.info(f"[{source_type.upper()}] {source} → {len(chunks)}개 청크")
            return {"status": "ok", "doc_id": doc_id, "chunks": len(chunks)}
        except ValueError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            logger.error(f"[{source_type.upper()}] {source} 실패: {e}")
            return {"status": "error", "message": str(e)}

    def ingest_batch(self, sources):
        results = {"success": 0, "fail": 0, "details": []}
        for src in sources:
            source_type = src.get("type", "")
            source = src.get("path") or src.get("url", "")
            result = self.ingest(source_type, source)
            results["success" if result["status"] == "ok" else "fail"] += 1
            results["details"].append(result)
        return results

    def get_stats(self):
        doc_count = self.conn.execute(
            "SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = self.conn.execute(
            "SELECT COUNT(*) FROM chunks").fetchone()[0]
        type_counts = {}
        for row in self.conn.execute(
                "SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type"):
            type_counts[row[0]] = row[1]
        return {"total_documents": doc_count, "total_chunks": chunk_count,
                "by_type": type_counts}


if __name__ == "__main__":
    rag = RAGSystem()
    sources = [
        {"type": "pdf", "path": "data/financial_report.txt"},
        {"type": "web", "url": "https://example.com/market-analysis"},
        {"type": "csv", "path": "data/portfolio.csv"},
    ]
    results = rag.ingest_batch(sources)
    print(f"성공: {results['success']}, 실패: {results['fail']}")
