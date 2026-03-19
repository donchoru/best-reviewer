import re
import requests
from .base import BaseLoader

DEFAULT_STRIP_TAGS = ("script", "style", "nav", "footer", "header")


class WebLoader(BaseLoader):

    def __init__(self, strip_tags: tuple[str, ...] = DEFAULT_STRIP_TAGS):
        self._strip_tags = strip_tags

    @property
    def source_type(self) -> str:
        return "web"

    def load(self, source: str) -> str:
        resp = requests.get(source, timeout=30, headers={"User-Agent": "RAGBot/1.0"})
        resp.raise_for_status()
        html = resp.text
        for tag in self._strip_tags:
            html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", html,
                          flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()
