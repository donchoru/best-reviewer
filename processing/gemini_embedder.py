"""Gemini 임베딩 — Text Embedding API 래퍼."""
import os
import requests
from .base_embedder import BaseEmbedder


class GeminiEmbedder(BaseEmbedder):

    def __init__(self, config):
        self._config = config
        self._api_key = os.environ.get(config.api_key_env, "")
        self._url = f"{config.api_url}/{config.model}:embedContent?key={self._api_key}"

    @property
    def dimension(self):
        return self._config.dimension

    def embed(self, text):
        resp = requests.post(self._url, json={
            "model": self._config.model,
            "content": {"parts": [{"text": text}]},
        }, timeout=self._config.timeout)
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]

    def embed_batch(self, texts):
        results = []
        for text in texts:
            try:
                results.append(self.embed(text))
            except Exception:
                results.append([0.0] * self.dimension)
        return results
