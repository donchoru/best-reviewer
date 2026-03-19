
from abc import ABC, abstractmethod


class BaseSimilarity(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def calculate(self, a: list[float], b: list[float]) -> float: ...


class CosineSimilarity(BaseSimilarity):

    @property
    def name(self) -> str:
        return "cosine"

    def calculate(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class EuclideanSimilarity(BaseSimilarity):

    @property
    def name(self) -> str:
        return "euclidean"

    def calculate(self, a, b):
        dist = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
        return 1.0 / (1.0 + dist)


class DotSimilarity(BaseSimilarity):

    @property
    def name(self) -> str:
        return "dot"

    def calculate(self, a, b):
        return sum(x * y for x, y in zip(a, b))


SIMILARITY_REGISTRY: dict[str, BaseSimilarity] = {
    "cosine": CosineSimilarity(),
    "euclidean": EuclideanSimilarity(),
    "dot": DotSimilarity(),
}


def get_similarity(name: str) -> BaseSimilarity:
    if name not in SIMILARITY_REGISTRY:
        raise ValueError(f"지원하지 않는 유사도: {name} "
                         f"(가능: {', '.join(SIMILARITY_REGISTRY)})")
    return SIMILARITY_REGISTRY[name]
