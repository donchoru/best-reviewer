import csv
from .base import BaseLoader


class CsvLoader(BaseLoader):

    @property
    def source_type(self) -> str:
        return "csv"

    def load(self, source: str) -> str:
        rows = []
        with open(source, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
        return "\n".join(rows)
