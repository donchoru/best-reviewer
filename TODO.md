# TODO — 추후 리팩토링 및 개선 사항

## Config 외부화
- [ ] ChunkConfig, EmbedConfig 등 설정값을 환경변수 또는 설정 파일(.env, yaml)에서 읽도록 변경
- [ ] config.py의 하드코딩된 기본값을 외부 설정 소스로 대체

## BaseStore ISP 분리
- [ ] BaseStore를 WriteStore, SearchStore, StatsStore로 인터페이스 분리
- [ ] 읽기 전용 replica 등 부분 구현이 가능하도록 개선

## GeminiEmbedder 에러 처리 개선
- [ ] embed_batch()의 silent zero vector 반환 대신 명시적 에러 전략 도입
- [ ] 재시도, 캐시 폴백 등 ErrorRecoveryStrategy 추상화 검토

## Pipeline 전처리 확장
- [ ] ingest() 내 텍스트 전처리 훅 추가 (정규화, PII 마스킹 등)
- [ ] TextProcessor 추상 인터페이스 도입 검토

## RAGSystem(rag_pipeline.py) DIP 개선
- [ ] 구체 클래스 직접 생성 대신 외부에서 주입받도록 변경
- [ ] main.create_pipeline()으로 완전 대체 후 레거시 래퍼 제거 검토
