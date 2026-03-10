"""설정 관리 — 매직 넘버 추출."""
from dataclasses import dataclass


@dataclass
class RAGConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    embed_model: str = "models/text-embedding-004"
    embed_api_url: str = "https://generativelanguage.googleapis.com/v1beta"
    embed_dimension: int = 768
    embed_timeout: int = 30
    api_key_env: str = "GEMINI_API_KEY"
    db_path: str = "/tmp/rag_assets.db"
    default_top_k: int = 5
