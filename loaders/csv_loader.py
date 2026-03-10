import csv
from .base import BaseLoader


class CsvLoader(BaseLoader):

    @property
    def source_type(self) -> str:
        return "csv"

    def load(self, source: str) -> str:
        rows = []
        with open(source, "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                return ""
            f.seek(0)
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
        return "\n".join(rows)
