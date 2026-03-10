"""Gemini 임베딩 — Text Embedding API 래퍼."""
import os
import requests


class GeminiEmbedder:

    def __init__(self, config):
        self._api_key = os.environ.get(config.api_key_env, "")
        self._url = (f"{config.embed_api_url}/{config.embed_model}"
                     f":embedContent?key={self._api_key}")
        self._model = config.embed_model
        self._timeout = config.embed_timeout
        self._dimension = config.embed_dimension

    @property
    def dimension(self):
        return self._dimension

    def embed(self, text):
        resp = requests.post(self._url, json={
            "model": self._model,
            "content": {"parts": [{"text": text}]},
        }, timeout=self._timeout)
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
