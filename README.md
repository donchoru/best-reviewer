# RAG Pipeline — 비정형 자산 수집

PDF, 웹 페이지, CSV 등 비정형 자산을 수집하여 벡터 임베딩으로 변환하고 검색하는 RAG 파이프라인.

## 구조

```
├── config.py           # RAGConfig — 설정 관리
├── document_loader.py  # DocumentLoader — 문서 로딩 (PDF/웹/CSV)
├── text_chunker.py     # TextChunker — 텍스트 청킹
├── embedder.py         # Embedder — Gemini 임베딩 생성
├── vector_store.py     # VectorStore — SQLite 벡터 저장소
├── pipeline.py         # RAGPipeline — 오케스트레이터
├── main.py             # 진입점
└── test_rag_pipeline.py  # 단위 테스트 (19건)
```

## 실행

```bash
python main.py
```

## 테스트

```bash
python -m unittest test_rag_pipeline -v
```

## 기술 스택

- Python 3.12+
- Gemini Text Embedding API (`text-embedding-004`)
- SQLite
