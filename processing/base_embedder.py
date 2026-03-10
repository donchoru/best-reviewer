"""임베더 추상 인터페이스."""
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):

    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...
