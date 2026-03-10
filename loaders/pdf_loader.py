from .base import BaseLoader


class PdfLoader(BaseLoader):

    @property
    def source_type(self) -> str:
        return "pdf"

    def load(self, source: str) -> str:
        # TODO: PDF 파서 연동 (현재는 텍스트 파일로 대체)
        with open(source, "r", encoding="utf-8") as f:
            return f.read()
