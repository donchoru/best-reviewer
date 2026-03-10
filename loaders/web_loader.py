import re
import requests
from .base import BaseLoader


class WebLoader(BaseLoader):

    @property
    def source_type(self) -> str:
        return "web"

    def load(self, source: str) -> str:
        resp = requests.get(source, timeout=30, headers={"User-Agent": "RAGBot/1.0"})
        resp.raise_for_status()
        html = resp.text
        for tag in ["script", "style", "nav", "footer", "header"]:
            html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", html,
                          flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", html)
        # TODO: JavaScript 렌더링 지원
        return re.sub(r"\s+", " ", text).strip()
