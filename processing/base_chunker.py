"""청커 추상 인터페이스 — Strategy 패턴."""
from abc import ABC, abstractmethod


class BaseChunker(ABC):

    @abstractmethod
    def split(self, text: str, doc_id: str, source: str, doc_type: str) -> list:
        """텍스트를 청크 리스트로 분할한다."""
