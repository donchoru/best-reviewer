"""설정 관리 — 관심사별 Nested Config."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkConfig:
    size: int = 1000
    overlap: int = 100


@dataclass(frozen=True)
class EmbedConfig:
    model: str = "models/text-embedding-004"
    api_url: str = "https://generativelanguage.googleapis.com/v1beta"
    dimension: int = 768
    timeout: int = 30
    api_key_env: str = "GEMINI_API_KEY"


@dataclass(frozen=True)
class StoreConfig:
    db_path: str = "/tmp/rag_assets.db"


@dataclass(frozen=True)
class RAGConfig:
    chunk: ChunkConfig = ChunkConfig()
    embed: EmbedConfig = EmbedConfig()
    store: StoreConfig = StoreConfig()
    default_top_k: int = 5
