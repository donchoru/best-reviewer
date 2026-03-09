# RAG Pipeline — 비정형 자산 수집

PDF, 웹 페이지, CSV 등 비정형 자산을 수집하여 벡터 임베딩으로 변환하고 검색하는 RAG 파이프라인.

## 구조

```
├── config.py                   # Nested Config (ChunkConfig, EmbedConfig, StoreConfig)
├── loaders/                    # Strategy 패턴 — 문서 로더
│   ├── base.py                 # BaseLoader (ABC)
│   ├── pdf_loader.py           # PDF 로더
│   ├── web_loader.py           # 웹 페이지 로더
│   └── csv_loader.py           # CSV 로더
├── processing/                 # 텍스트 처리
│   ├── chunker.py              # TextChunker
│   ├── base_embedder.py        # BaseEmbedder (ABC)
│   └── gemini_embedder.py      # Gemini 임베딩 구현체
├── stores/                     # Repository 패턴 — 벡터 저장소
│   ├── base_store.py           # BaseStore (ABC)
│   └── sqlite_store.py         # SQLite 구현체
├── pipeline.py                 # LoaderRegistry + RAGPipeline (DI)
└── main.py                     # Composition Root (의존성 조립)
```

## 설계 원칙

- **SRP**: 각 클래스가 단일 책임만 담당
- **OCP**: 새 로더 추가 = 새 파일 1개 + `registry.register()` 1줄
- **DIP**: `RAGPipeline`은 추상 인터페이스(`BaseLoader`, `BaseEmbedder`, `BaseStore`)에만 의존
- **Strategy**: 로더/임베더를 런타임에 교체 가능
- **Repository**: 저장소 구현을 인터페이스 뒤에 캡슐화

## 실행

```bash
python main.py
```

## 확장 예시 — JSON 로더 추가

```python
# loaders/json_loader.py (새 파일)
from .base import BaseLoader

class JsonLoader(BaseLoader):
    @property
    def source_type(self) -> str:
        return "json"

    def load(self, source: str) -> str:
        ...
```

```python
# main.py에 1줄 추가
registry.register(JsonLoader())
```

기존 코드 수정: **0줄**.

## 기술 스택

- Python 3.12+
- Gemini Text Embedding API (`text-embedding-004`)
- SQLite
