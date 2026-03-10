"""설정 관리 — 관심사별 Config 분리."""
from dataclasses import dataclass


@dataclass
class StoreConfig:
    db_path: str = "/tmp/rag_assets.db"


@dataclass
class RAGConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    embed_model: str = "models/text-embedding-004"
    embed_api_url: str = "https://generativelanguage.googleapis.com/v1beta"
    embed_dimension: int = 768
    embed_timeout: int = 30
    api_key_env: str = "GEMINI_API_KEY"
    store: StoreConfig = None
    default_top_k: int = 5

    def __post_init__(self):
        if self.store is None:
            self.store = StoreConfig()
