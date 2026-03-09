import json
import hashlib
import sqlite3
from datetime import datetime
from config import StoreConfig
from processing.chunker import Chunk
from .base_store import BaseStore


class SqliteVectorStore(BaseStore):

    def __init__(self, config: StoreConfig):
        self._conn = sqlite3.connect(config.db_path)
        self._init_tables()

    def _init_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, source TEXT, content TEXT,
                doc_type TEXT, created_at TEXT, chunk_count INTEGER, status TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, doc_id TEXT, content TEXT,
                embedding TEXT, position INTEGER
            )
        """)
        self._conn.commit()

    def save_document(self, doc_id: str, source: str, content: str,
                      doc_type: str, chunk_count: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, source, content[:1000], doc_type,
             datetime.now().isoformat(), chunk_count, "processed"),
        )
        self._conn.commit()

    def save_chunks(self, chunks: list[Chunk],
                    embeddings: list[list[float]]) -> None:
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = hashlib.md5(chunk.content.encode()).hexdigest()
            self._conn.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)",
                (chunk_id, chunk.doc_id, chunk.content,
                 json.dumps(embedding), chunk.position),
            )
        self._conn.commit()

    def search_similar(self, query_embedding: list[float],
                       top_k: int) -> list[dict]:
        cursor = self._conn.execute(
            "SELECT id, doc_id, content, embedding FROM chunks")
        results = []
        for row in cursor:
            chunk_emb = json.loads(row[3])
            if not chunk_emb or all(v == 0.0 for v in chunk_emb):
                continue
            score = self._cosine_similarity(query_embedding, chunk_emb)
            results.append({"chunk_id": row[0], "doc_id": row[1],
                            "content": row[2], "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_stats(self) -> dict:
        doc_count = self._conn.execute(
            "SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = self._conn.execute(
            "SELECT COUNT(*) FROM chunks").fetchone()[0]
        type_counts = {}
        for row in self._conn.execute(
                "SELECT doc_type, COUNT(*) FROM documents GROUP BY doc_type"):
            type_counts[row[0]] = row[1]
        return {"total_documents": doc_count, "total_chunks": chunk_count,
                "by_type": type_counts}

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
