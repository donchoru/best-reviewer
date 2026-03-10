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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG 파이프라인: 문서 로딩 → 청킹 → 임베딩 → 저장 → 검색."""

    def __init__(self, db_path="/tmp/rag_assets.db"):
        self.db_path = db_path
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.conn = sqlite3.connect(db_path)
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

    # ── 문서 로딩 ──────────────────────────────────────────

    def load_pdf(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"PDF 로딩 실패: {e}")
            raise

    def load_web(self, url):
        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "RAGBot/1.0"})
            resp.raise_for_status()
            html = resp.text
            for tag in ["script", "style", "nav", "footer", "header"]:
                html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", html,
                              flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", html)
            return re.sub(r"\s+", " ", text).strip()
        except Exception as e:
            logger.error(f"웹 로딩 실패: {e}")
            raise

    def load_csv(self, path):
        try:
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
            return "\n".join(rows)
        except Exception as e:
            logger.error(f"CSV 로딩 실패: {e}")
            raise

    # ── 텍스트 청킹 ────────────────────────────────────────

    def chunk_text(self, text, doc_id, source, doc_type):
        chunks = []
        start = 0
        while start < len(text):
            segment = text[start:start + 500]
            if segment.strip():
                chunks.append({
                    "content": segment,
                    "position": len(chunks),
                    "doc_id": doc_id,
                    "source": source,
                    "doc_type": doc_type,
                })
            start += 500 - 50
        return chunks

    # ── 임베딩 ─────────────────────────────────────────────

    def get_embedding(self, text):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={self.api_key}"
        resp = requests.post(url, json={
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text}]},
        }, timeout=30)
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]

    def get_embeddings_batch(self, texts):
        results = []
        for text in texts:
            try:
                results.append(self.get_embedding(text))
            except Exception:
                results.append([0.0] * 768)
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
            chunk_id = hashlib.md5(chunk["content"].encode()).hexdigest()
            self.conn.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)",
                (chunk_id, chunk["doc_id"], chunk["content"],
                 json.dumps(embedding), chunk["position"]),
            )
        self.conn.commit()

    # ── 검색 ───────────────────────────────────────────────

    def calc_similarity(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def search(self, query, top_k=5):
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
        return results[:top_k]

    # ── 파이프라인 ─────────────────────────────────────────

    def ingest(self, source_type, source):
        try:
            if source_type == "pdf":
                text = self.load_pdf(source)
            elif source_type == "web":
                text = self.load_web(source)
            elif source_type == "csv":
                text = self.load_csv(source)
            else:
                return {"status": "error", "message": f"지원하지 않는 타입: {source_type}"}

            if not text.strip():
                return {"status": "error", "message": "빈 콘텐츠"}

            doc_id = hashlib.md5(text.encode()).hexdigest()
            chunks = self.chunk_text(text, doc_id, source, source_type)
            embeddings = self.get_embeddings_batch([c["content"] for c in chunks])

            self.save_document(doc_id, source, text, source_type, len(chunks))
            self.save_chunks(chunks, embeddings)

            logger.info(f"[{source_type.upper()}] {source} → {len(chunks)}개 청크")
            return {"status": "ok", "doc_id": doc_id, "chunks": len(chunks)}
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
