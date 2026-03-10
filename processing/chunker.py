"""텍스트 청킹 — 문서를 검색 가능한 단위로 분할."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    content: str
    position: int
    doc_id: str
    source: str
    doc_type: str


class TextChunker:

    def __init__(self, config):
        self._size = config.size
        self._overlap = config.overlap

    def split(self, text, doc_id, source, doc_type):
        """텍스트를 고정 크기 청크로 분할한다."""
        chunks = []
        start = 0
        while start < len(text):
            segment = text[start:start + self._size]
            if segment.strip():
                chunks.append(Chunk(segment, len(chunks), doc_id, source, doc_type))
            start += self._size - self._overlap
        return chunks
