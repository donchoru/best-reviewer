"""텍스트 청킹 — 중복 제거된 단일 청킹 로직."""
from dataclasses import dataclass
from config import RAGConfig


@dataclass
class Chunk:
    """청크 데이터."""
    content: str
    position: int
    doc_id: str
    source: str
    doc_type: str


class TextChunker:
    """텍스트를 고정 크기 청크로 분할한다."""

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
                chunks.append(Chunk(
                    content=segment,
                    position=len(chunks),
                    doc_id=doc_id,
                    source=source,
                    doc_type=doc_type,
                ))
            start = end - self.overlap
        return chunks
