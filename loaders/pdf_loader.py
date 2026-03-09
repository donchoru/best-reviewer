"""PDF 로더."""
from .base import BaseLoader


class PdfLoader(BaseLoader):

    @property
    def source_type(self) -> str:
        return "pdf"

    def load(self, source: str) -> str:
        with open(source, "r", encoding="utf-8") as f:
            return f.read()
