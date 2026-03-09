"""설정 관리 — Magic Number/String 제거, 단일 설정 소스."""
from dataclasses import dataclass


@dataclass
class RAGConfig:
    """RAG 파이프라인 설정."""
    # 청킹
    chunk_size: int = 500
    chunk_overlap: int = 50

    # 임베딩
    embed_model: str = "models/text-embedding-004"
    embed_api_url: str = "https://generativelanguage.googleapis.com/v1beta"
    embed_dimension: int = 768
    embed_timeout: int = 30

    # DB
    db_path: str = "/tmp/rag_assets.db"

    # 검색
    default_top_k: int = 5

    # API 키 환경변수명
    api_key_env: str = "GEMINI_API_KEY"
