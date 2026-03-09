# RAG Pipeline — 비정형 자산 수집

PDF, 웹 페이지, CSV 등 비정형 자산을 수집하여 벡터 임베딩으로 변환하고 검색하는 RAG 파이프라인.

## 구조

```
├── config.py               # Nested Config (ChunkConfig, EmbedConfig, StoreConfig)
├── loaders/                 # Strategy 패턴 — 로더 확장 자유
│   ├── base.py              # BaseLoader ABC
│   ├── pdf_loader.py        # PdfLoader
│   ├── web_loader.py        # WebLoader
│   └── csv_loader.py        # CsvLoader
├── processing/              # 텍스트 처리
│   ├── base_embedder.py     # BaseEmbedder ABC
│   ├── gemini_embedder.py   # GeminiEmbedder (구현)
│   └── chunker.py           # TextChunker + Chunk (frozen)
├── stores/                  # Repository 패턴 — 저장소 교체 자유
│   ├── base_store.py        # BaseStore ABC
│   └── sqlite_store.py      # SqliteVectorStore (구현)
├── pipeline.py              # LoaderRegistry (OCP) + RAGPipeline (DIP)
├── main.py                  # Composition Root — DI 조립
└── test_rag_pipeline.py     # 단위 테스트 (31건)
```

## 적용 원칙

- **SRP** — Config / Loader / Chunker / Embedder / Store / Pipeline 각각 단일 책임
- **OCP** — LoaderRegistry에 새 로더 등록만으로 확장 (기존 코드 수정 불필요)
- **LSP** — BaseLoader를 구현한 어떤 로더든 교체 가능
- **ISP** — BaseEmbedder, BaseStore 각각 필요한 메서드만 정의
- **DIP** — RAGPipeline이 구체 클래스가 아닌 추상 인터페이스에 의존

## 실행

```bash
python main.py
```

## 테스트

```bash
python -m pytest test_rag_pipeline.py -v
```

## 기술 스택

- Python 3.12+
- Gemini Text Embedding API (`text-embedding-004`)
- SQLite
