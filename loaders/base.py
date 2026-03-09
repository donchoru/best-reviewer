"""문서 로더 추상 인터페이스 — Strategy 패턴."""
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """문서 로더 인터페이스.

    새 문서 타입 추가 시 이 클래스를 상속하고
    load()만 구현하면 된다. (OCP)
    """

    @abstractmethod
    def load(self, source: str) -> str:
        """소스에서 텍스트를 추출한다."""

    @property
    @abstractmethod
    def source_type(self) -> str:
        """이 로더가 처리하는 소스 타입 식별자."""
