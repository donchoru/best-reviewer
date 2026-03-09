"""문서 로딩 — PDF/웹/CSV 로딩을 단일 클래스로 통합."""
import csv
import re
import requests


class DocumentLoader:

    def load_pdf(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def load_web(self, url: str) -> str:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "RAGBot/1.0"})
        resp.raise_for_status()
        html = resp.text
        for tag in ["script", "style", "nav", "footer", "header"]:
            html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", html,
                          flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()

    def load_csv(self, file_path: str) -> str:
        rows = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
        return "\n".join(rows)

    def load(self, source_type: str, source: str) -> str:
        loaders = {"pdf": self.load_pdf, "web": self.load_web, "csv": self.load_csv}
        loader = loaders.get(source_type)
        if not loader:
            raise ValueError(f"지원하지 않는 소스 타입: {source_type}")
        return loader(source)
