"""임베딩 생성 — 4회 반복되던 API 호출 로직을 단일 클래스로 통합."""
import os
import requests
from config import RAGConfig


class Embedder:

    def __init__(self, config: RAGConfig):
        self.config = config
        self.api_key = os.environ.get(config.api_key_env, "")
        self.url = (f"{config.embed_api_url}/{config.embed_model}"
                    f":embedContent?key={self.api_key}")

    def embed(self, text: str) -> list[float]:
        resp = requests.post(self.url, json={
            "model": self.config.embed_model,
            "content": {"parts": [{"text": text}]},
        }, timeout=self.config.embed_timeout)
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            try:
                results.append(self.embed(text))
            except Exception:
                results.append([0.0] * self.config.embed_dimension)
        return results
