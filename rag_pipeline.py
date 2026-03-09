"""비정형 자산 수집 RAG 파이프라인."""
import os
import re
import csv
import json
import hashlib
import sqlite3
import requests
from datetime import datetime


# 전역 상태
processed_count = 0
error_log = []
db_connection = None


class RAGSystem:
    """RAG 파이프라인 전체를 담당하는 클래스."""

    def __init__(self):
        self.documents = []
        self.chunks = []
        self.embeddings = []
        self.collection_name = "assets"
        self.embed_model = "models/text-embedding-004"
        self.db_path = "/tmp/rag_assets.db"
        self._init_db()

    def _init_db(self):
        global db_connection
        db_connection = sqlite3.connect(self.db_path)
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

    # ── PDF 처리 ──────────────────────────────────────────────
    def process_pdf(self, file_path: str) -> dict:
        global processed_count, error_log

        print(f"[PDF] 처리 시작: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            error_log.append(f"PDF 읽기 실패: {file_path} - {e}")
            print(f"[ERROR] PDF 읽기 실패: {e}")
            return {"status": "error", "message": str(e)}

        if len(text) == 0:
            error_log.append(f"빈 파일: {file_path}")
            return {"status": "error", "message": "빈 파일"}

        doc_id = hashlib.md5(text.encode()).hexdigest()

        # 청킹
        chunks = []
        start = 0
        while start < len(text):
            end = start + 500
            chunk = text[start:end]
            if len(chunk.strip()) > 0:
                chunks.append({
                    "content": chunk, "position": len(chunks),
                    "doc_id": doc_id, "source": file_path, "doc_type": "pdf",
                })
            start = end - 50

        # 임베딩
        for chunk in chunks:
            try:
                api_key = os.environ.get("GEMINI_API_KEY", "")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
                resp = requests.post(url, json={
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": chunk["content"]}]}
                }, timeout=30)
                if resp.status_code == 200:
                    chunk["embedding"] = resp.json()["embedding"]["values"]
                else:
                    error_log.append(f"임베딩 실패: {resp.status_code}")
                    chunk["embedding"] = [0.0] * 768
            except Exception as e:
                error_log.append(f"임베딩 에러: {e}")
                chunk["embedding"] = [0.0] * 768

        # DB 저장
        global db_connection
        db_connection.execute(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, file_path, text[:1000], "pdf",
             datetime.now().isoformat(), len(chunks), "processed")
        )
        for chunk in chunks:
            chunk_id = hashlib.md5(chunk["content"].encode()).hexdigest()
            db_connection.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)",
                (chunk_id, doc_id, chunk["content"],
                 json.dumps(chunk.get("embedding", [])), chunk["position"])
            )
        db_connection.commit()

        processed_count += 1
        print(f"[PDF] 완료: {len(chunks)}개 청크 생성")
        return {"status": "ok", "doc_id": doc_id, "chunks": len(chunks)}

    # ── 웹 페이지 처리 ────────────────────────────────────────
    def process_web(self, url: str) -> dict:
        global processed_count, error_log

        print(f"[WEB] 처리 시작: {url}")
        try:
            resp = requests.get(url, timeout=30,
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
            print(f"[ERROR] 웹 크롤링 실패: {e}")
            return {"status": "error", "message": str(e)}

        if len(text) == 0:
            error_log.append(f"빈 페이지: {url}")
            return {"status": "error", "message": "빈 페이지"}

        doc_id = hashlib.md5(text.encode()).hexdigest()

        # 청킹 (process_pdf와 동일한 코드)
        chunks = []
        start = 0
        while start < len(text):
            end = start + 500
            chunk = text[start:end]
            if len(chunk.strip()) > 0:
                chunks.append({
                    "content": chunk, "position": len(chunks),
                    "doc_id": doc_id, "source": url, "doc_type": "web",
                })
            start = end - 50

        # 임베딩 (process_pdf와 동일한 코드)
        for chunk in chunks:
            try:
                api_key = os.environ.get("GEMINI_API_KEY", "")
                embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
                resp = requests.post(embed_url, json={
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": chunk["content"]}]}
                }, timeout=30)
                if resp.status_code == 200:
                    chunk["embedding"] = resp.json()["embedding"]["values"]
                else:
                    error_log.append(f"임베딩 실패: {resp.status_code}")
                    chunk["embedding"] = [0.0] * 768
            except Exception as e:
                error_log.append(f"임베딩 에러: {e}")
                chunk["embedding"] = [0.0] * 768

        # DB 저장 (process_pdf와 동일한 코드)
        global db_connection
        db_connection.execute(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, url, text[:1000], "web",
             datetime.now().isoformat(), len(chunks), "processed")
        )
        for chunk in chunks:
            chunk_id = hashlib.md5(chunk["content"].encode()).hexdigest()
            db_connection.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)",
                (chunk_id, doc_id, chunk["content"],
                 json.dumps(chunk.get("embedding", [])), chunk["position"])
            )
        db_connection.commit()

        processed_count += 1
        print(f"[WEB] 완료: {len(chunks)}개 청크 생성")
        return {"status": "ok", "doc_id": doc_id, "chunks": len(chunks)}

    # ── CSV 처리 ──────────────────────────────────────────────
    def process_csv(self, file_path: str) -> dict:
        global processed_count, error_log

        print(f"[CSV] 처리 시작: {file_path}")
        try:
            rows = []
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
            text = "\n".join(rows)
        except Exception as e:
            error_log.append(f"CSV 읽기 실패: {file_path} - {e}")
            print(f"[ERROR] CSV 읽기 실패: {e}")
            return {"status": "error", "message": str(e)}

        if len(text) == 0:
            error_log.append(f"빈 CSV: {file_path}")
            return {"status": "error", "message": "빈 CSV"}

        doc_id = hashlib.md5(text.encode()).hexdigest()

        # 청킹 (3번째 반복)
        chunks = []
        start = 0
        while start < len(text):
            end = start + 500
            chunk = text[start:end]
            if len(chunk.strip()) > 0:
                chunks.append({
                    "content": chunk, "position": len(chunks),
                    "doc_id": doc_id, "source": file_path, "doc_type": "csv",
                })
            start = end - 50

        # 임베딩 (3번째 반복)
        for chunk in chunks:
            try:
                api_key = os.environ.get("GEMINI_API_KEY", "")
                embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
                resp = requests.post(embed_url, json={
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": chunk["content"]}]}
                }, timeout=30)
                if resp.status_code == 200:
                    chunk["embedding"] = resp.json()["embedding"]["values"]
                else:
                    error_log.append(f"임베딩 실패: {resp.status_code}")
                    chunk["embedding"] = [0.0] * 768
            except Exception as e:
                error_log.append(f"임베딩 에러: {e}")
                chunk["embedding"] = [0.0] * 768

        # DB 저장 (3번째 반복)
        global db_connection
        db_connection.execute(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, file_path, text[:1000], "csv",
             datetime.now().isoformat(), len(chunks), "processed")
        )
        for chunk in chunks:
            chunk_id = hashlib.md5(chunk["content"].encode()).hexdigest()
            db_connection.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)",
                (chunk_id, doc_id, chunk["content"],
                 json.dumps(chunk.get("embedding", [])), chunk["position"])
            )
        db_connection.commit()

        processed_count += 1
        print(f"[CSV] 완료: {len(chunks)}개 청크 생성")
        return {"status": "ok", "doc_id": doc_id, "chunks": len(chunks)}

    # ── 검색 ──────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
        try:
            resp = requests.post(url, json={
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": query}]}
            }, timeout=30)
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
        return results[:top_k]

    # ── 통계 ──────────────────────────────────────────────────
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
            "total_documents": doc_count,
            "total_chunks": chunk_count,
            "by_type": type_counts,
            "processed_count": processed_count,
            "error_count": len(error_log),
            "errors": error_log[-10:],
        }

    # ── 미사용 메서드들 ───────────────────────────────────────
    def export_to_json(self, path: str):
        """JSON으로 내보내기 — 어디서도 호출하지 않음."""
        global db_connection
        docs = db_connection.execute("SELECT * FROM documents").fetchall()
        with open(path, "w") as f:
            json.dump([{"id": d[0], "source": d[1], "type": d[3]}
                       for d in docs], f)

    def reindex_all(self):
        """전체 재인덱싱 — 미완성 상태로 방치."""
        pass

    def validate_embeddings(self):
        """임베딩 유효성 검사 — 호출하는 곳 없음."""
        global db_connection
        cursor = db_connection.execute("SELECT id, embedding FROM chunks")
        invalid = 0
        for row in cursor:
            emb = json.loads(row[1])
            if len(emb) != 768:
                invalid += 1
        return invalid

    # ── 일괄 처리 ─────────────────────────────────────────────
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
                print(f"[WARN] 지원하지 않는 타입: {source.get('type')}")
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

        print(f"\n{'='*50}")
        print(f"처리 완료: 성공 {results['success']}, 실패 {results['fail']}")
        print(f"{'='*50}")
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

    results = rag.search("비정형 자산 포트폴리오 분석")
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
