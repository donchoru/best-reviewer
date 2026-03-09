"""텍스트 청킹."""
from dataclasses import dataclass
from config import ChunkConfig


@dataclass(frozen=True)
class Chunk:
    content: str
    position: int
    doc_id: str
    source: str
    doc_type: str


class TextChunker:
    """고정 크기 + 오버랩 청킹."""

    def __init__(self, config: ChunkConfig):
        self._size = config.size
        self._overlap = config.overlap

    def split(self, text: str, doc_id: str, source: str, doc_type: str) -> list[Chunk]:
        chunks = []
        start = 0
        while start < len(text):
            segment = text[start:start + self._size]
            if segment.strip():
                chunks.append(Chunk(segment, len(chunks), doc_id, source, doc_type))
            start += self._size - self._overlap
        return chunks
