# RAG Pipeline — 비정형 자산 수집

PDF, 웹 페이지, CSV 등 비정형 자산을 수집하여 벡터 임베딩으로 변환하고 검색하는 RAG 파이프라인.

## 기능

- PDF/웹/CSV 문서를 텍스트로 변환
- 고정 크기 청킹 (500자, 50자 오버랩)
- Gemini Text Embedding API로 벡터 임베딩 생성
- SQLite에 문서/청크/임베딩 저장
- 코사인 유사도 기반 검색

## 실행

```bash
python rag_pipeline.py
```

## 기술 스택

- Python 3.12+
- Gemini Text Embedding API (`text-embedding-004`)
- SQLite
