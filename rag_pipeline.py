"""비정형 자산 수집 RAG 파이프라인.

Step 1: RAGConfig 추출 — 매직 넘버를 설정 객체로 교체.
아직 중복 코드(청킹/임베딩/DB저장)는 process_pdf/web/csv에 남아있다.
"""
import os
import re
import csv
import json
import hashlib
import sqlite3
import requests
from datetime import datetime
from config import RAGConfig


processed_count = 0
error_log = []
db_connection = None


class RAGSystem:

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self.documents = []
        self.chunks = []
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

    def process_pdf(self, file_path: str) -> dict:
        global processed_count, error_log

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            error_log.append(f"PDF 읽기 실패: {file_path} - {e}")
            return {"status": "error", "message": str(e)}

        if len(text) == 0:
            return {"status": "error", "message": "빈 파일"}

        doc_id = hashlib.md5(text.encode()).hexdigest()

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.config.chunk_size
            chunk = text[start:end]
            if len(chunk.strip()) > 0:
                chunks.append({
                    "content": chunk, "position": len(chunks),
                    "doc_id": doc_id, "source": file_path, "doc_type": "pdf",
                })
            start = end - self.config.chunk_overlap

        api_key = os.environ.get(self.config.api_key_env, "")
        url = f"{self.config.embed_api_url}/{self.config.embed_model}:embedContent?key={api_key}"
        for chunk in chunks:
            try:
                resp = requests.post(url, json={
                    "model": self.config.embed_model,
                    "content": {"parts": [{"text": chunk["content"]}]}
                }, timeout=self.config.embed_timeout)
                if resp.status_code == 200:
                    chunk["embedding"] = resp.json()["embedding"]["values"]
                else:
                    chunk["embedding"] = [0.0] * self.config.embed_dimension
            except Exception as e:
                error_log.append(f"임베딩 에러: {e}")
                chunk["embedding"] = [0.0] * self.config.embed_dimension

        global db_connection
        db_connection.execute(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, file_path, text[:1000], "pdf",
             datetime.now().isoformat(), len(chunks), "processed"))
        for chunk in chunks:
            chunk_id = hashlib.md5(chunk["content"].encode()).hexdigest()
            db_connection.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)",
                (chunk_id, doc_id, chunk["content"],
                 json.dumps(chunk.get("embedding", [])), chunk["position"]))
        db_connection.commit()

        processed_count += 1
        return {"status": "ok", "doc_id": doc_id, "chunks": len(chunks)}

    def process_web(self, url: str) -> dict:
        global processed_count, error_log

        try:
            resp = requests.get(url, timeout=self.config.embed_timeout,
                                headers={"User-Agent": "RAGBot/1.0"})
            resp.raise_for_status()
            text = resp.text
            for tag in ["script", "style", "nav", "footer", "header"]:
                text = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", text,
                              flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
        except Exception as e:
            error_log.append(f"웹 크롤링 실패: {url} - {e}")
            return {"status": "error", "message": str(e)}

        if len(text) == 0:
            return {"status": "error", "message": "빈 페이지"}

        doc_id = hashlib.md5(text.encode()).hexdigest()

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.config.chunk_size
            chunk = text[start:end]
            if len(chunk.strip()) > 0:
                chunks.append({
                    "content": chunk, "position": len(chunks),
                    "doc_id": doc_id, "source": url, "doc_type": "web",
                })
            start = end - self.config.chunk_overlap

        api_key = os.environ.get(self.config.api_key_env, "")
        embed_url = f"{self.config.embed_api_url}/{self.config.embed_model}:embedContent?key={api_key}"
        for chunk in chunks:
            try:
                resp = requests.post(embed_url, json={
                    "model": self.config.embed_model,
                    "content": {"parts": [{"text": chunk["content"]}]}
                }, timeout=self.config.embed_timeout)
                if resp.status_code == 200:
                    chunk["embedding"] = resp.json()["embedding"]["values"]
                else:
                    chunk["embedding"] = [0.0] * self.config.embed_dimension
            except Exception as e:
                error_log.append(f"임베딩 에러: {e}")
                chunk["embedding"] = [0.0] * self.config.embed_dimension

        global db_connection
        db_connection.execute(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, url, text[:1000], "web",
             datetime.now().isoformat(), len(chunks), "processed"))
        for chunk in chunks:
            chunk_id = hashlib.md5(chunk["content"].encode()).hexdigest()
            db_connection.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)",
                (chunk_id, doc_id, chunk["content"],
                 json.dumps(chunk.get("embedding", [])), chunk["position"]))
        db_connection.commit()

        processed_count += 1
        return {"status": "ok", "doc_id": doc_id, "chunks": len(chunks)}

    def process_csv(self, file_path: str) -> dict:
        global processed_count, error_log

        try:
            rows = []
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
            text = "\n".join(rows)
        except Exception as e:
            error_log.append(f"CSV 읽기 실패: {file_path} - {e}")
            return {"status": "error", "message": str(e)}

        if len(text) == 0:
            return {"status": "error", "message": "빈 CSV"}

        doc_id = hashlib.md5(text.encode()).hexdigest()

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.config.chunk_size
            chunk = text[start:end]
            if len(chunk.strip()) > 0:
                chunks.append({
                    "content": chunk, "position": len(chunks),
                    "doc_id": doc_id, "source": file_path, "doc_type": "csv",
                })
            start = end - self.config.chunk_overlap

        api_key = os.environ.get(self.config.api_key_env, "")
        embed_url = f"{self.config.embed_api_url}/{self.config.embed_model}:embedContent?key={api_key}"
        for chunk in chunks:
            try:
                resp = requests.post(embed_url, json={
                    "model": self.config.embed_model,
                    "content": {"parts": [{"text": chunk["content"]}]}
                }, timeout=self.config.embed_timeout)
                if resp.status_code == 200:
                    chunk["embedding"] = resp.json()["embedding"]["values"]
                else:
                    chunk["embedding"] = [0.0] * self.config.embed_dimension
            except Exception as e:
                error_log.append(f"임베딩 에러: {e}")
                chunk["embedding"] = [0.0] * self.config.embed_dimension

        global db_connection
        db_connection.execute(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, file_path, text[:1000], "csv",
             datetime.now().isoformat(), len(chunks), "processed"))
        for chunk in chunks:
            chunk_id = hashlib.md5(chunk["content"].encode()).hexdigest()
            db_connection.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)",
                (chunk_id, doc_id, chunk["content"],
                 json.dumps(chunk.get("embedding", [])), chunk["position"]))
        db_connection.commit()

        processed_count += 1
        return {"status": "ok", "doc_id": doc_id, "chunks": len(chunks)}

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.config.default_top_k
        api_key = os.environ.get(self.config.api_key_env, "")
        url = f"{self.config.embed_api_url}/{self.config.embed_model}:embedContent?key={api_key}"
        try:
            resp = requests.post(url, json={
                "model": self.config.embed_model,
                "content": {"parts": [{"text": query}]}
            }, timeout=self.config.embed_timeout)
            query_embedding = resp.json()["embedding"]["values"]
        except Exception:
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

    def export_to_json(self, path: str):
        global db_connection
        docs = db_connection.execute("SELECT * FROM documents").fetchall()
        with open(path, "w") as f:
            json.dump([{"id": d[0], "source": d[1], "type": d[3]}
                       for d in docs], f)

    def reindex_all(self):
        pass

    def validate_embeddings(self):
        global db_connection
        cursor = db_connection.execute("SELECT id, embedding FROM chunks")
        invalid = 0
        for row in cursor:
            emb = json.loads(row[1])
            if len(emb) != self.config.embed_dimension:
                invalid += 1
        return invalid

    def process_all(self, sources: list[dict]) -> dict:
        results = {"success": 0, "fail": 0, "details": []}
        for source in sources:
            if source.get("type") == "pdf":
                result = self.process_pdf(source["path"])
            elif source.get("type") == "web":
                result = self.process_web(source["url"])
            elif source.get("type") == "csv":
                result = self.process_csv(source["path"])
            else:
                results["fail"] += 1
                results["details"].append({
                    "status": "error",
                    "message": f"지원하지 않는 타입: {source.get('type')}"
                })
                continue

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
