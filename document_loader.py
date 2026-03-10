"""문서 로딩 — PDF, Web, CSV 소스에서 텍스트 추출."""
import re
import csv
import requests
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """소스 타입별 문서 로딩을 담당한다."""

    def load(self, source_type, source):
        """소스 타입에 따라 적절한 로더를 호출한다."""
        if source_type == "pdf":
            return self._load_pdf(source)
        elif source_type == "web":
            return self._load_web(source)
        elif source_type == "csv":
            return self._load_csv(source)
        else:
            raise ValueError(f"지원하지 않는 타입: {source_type}")

    def _load_pdf(self, path):
        # TODO: PDF 파서 연동 (현재는 텍스트 파일로 대체)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _load_web(self, url):
        resp = requests.get(url, timeout=30, headers={"User-Agent": "RAGBot/1.0"})
        resp.raise_for_status()
        html = resp.text
        for tag in ["script", "style", "nav", "footer", "header"]:
            html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", html,
                          flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", html)
        # TODO: JavaScript 렌더링 지원
        return re.sub(r"\s+", " ", text).strip()

    def _load_csv(self, path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
        return "\n".join(rows)
