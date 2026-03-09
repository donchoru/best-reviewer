"""텍스트 청킹 — 3회 중복되던 로직을 단일 구현으로 통합."""
from dataclasses import dataclass
from config import RAGConfig


@dataclass
class Chunk:
    content: str
    position: int
    doc_id: str
    source: str
    doc_type: str


class TextChunker:

    def __init__(self, config: RAGConfig):
        self.chunk_size = config.chunk_size
        self.overlap = config.chunk_overlap

    def split(self, text: str, doc_id: str, source: str, doc_type: str) -> list[Chunk]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            segment = text[start:end]
            if segment.strip():
                chunks.append(Chunk(segment, len(chunks), doc_id, source, doc_type))
            start = end - self.overlap
        return chunks
