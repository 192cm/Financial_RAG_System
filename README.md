# 📈 지능형 멀티모달 금융 RAG 시스템
### (Intelligent Multi-modal Financial RAG with Vision Self-Correction Agent)

본 프로젝트는 금융감독원 전자공시시스템(DART)의 방대한 비정형/멀티모달 사업보고서(PDF)에서 사용자가 원하는 재무 데이터를 왜곡 없이 정밀하게 추출하기 위해 설계된 **전문가용 하이브리드 RAG 시스템**입니다.

---

## ✨ 핵심 기술 및 특징 (Core Technologies)

1. **ColPali v1.2 기반 비전 검색**: 텍스트 추출의 한계를 넘어 이미지(표, 차트) 자체를 벡터화하여 검색하는 최신 시각적 리트리벌 방식 도입.
2. **자가 교정(Corrective) 에이전트**: Gemini 모델이 검색된 이미지를 직접 읽고 질문과의 정합성을 스스로 검증하며, 오답 시 후보군을 확장하는 루프 구현.
3. **하이브리드 앙상블 검색**: 고유명사에 강한 BM25와 맥락에 강한 Vector 검색을 결합하여 금융 전문 용어 검색의 정확도 극대화.

---

## 📂 상세 프로젝트 구조 (Detailed Directory Structure)

```text
Financial_RAG_System/
│
├── src/                            # 🛠️ 핵심 고속 엔진 모듈
│   ├── retrieval/                  # 검색 엔진 레이어
│   │   ├── vision_engine.py        # ColPali 기반 비전 인덱싱 및 검색 (새로운 index() 기능 포함)
│   │   ├── text_engine.py          # BM25 + ParentDocumentRetriever 하이브리드 검색
│   │   └── router.py               # 쿼리 기반 타겟 기업 자동 추론 및 필터링
│   │
│   ├── engines/                    # 실행 엔진 레이어
│   │   ├── rag_engines.py          # Method 0-3의 모든 RAG 워크플로우 통합 관리
│   │   └── agent.py                # 가이딩된 반복형(Iterative) 자가 교정 에이전트
│   │
│   ├── evaluation/                 # 평가 자동화 레이어
│   │   ├── runner.py               # 벤치마크 실행 및 API 에이전트 방어 로직 (Retry/Backoff)
│   │   ├── judge.py                # LLM-as-a-Judge 기반 정성 평가 자동화
│   │   └── metrics.py              # EM, ROUGE-L, BLEU 등 정량 메트릭 계산
│   │
│   ├── utils/                      # 공용 유틸리티 레이어
│   │   ├── vision.py               # 이미지/PDF 처리 유틸리티
│   │   └── common.py               # [NEW] 응답 정제(clean_llm_response) 및 공통 유틸
│   │
│   ├── config.py                   # 중앙 설정 관리 (Base Dir, DB 경로, API 환경변수 등)
│   ├── models.py                   # LLM 팩토리 (Gemini 최적화 설정 - max_retries, timeout)
│   └── __init__.py                 # 패키지 인터페이스 정의
│
├── notebooks/                      # 📓 실험 및 결과 시각화
│   ├── 01_baseline_rag.ipynb       # 텍스트 단독 RAG 성능 점검
│   ├── 02_method1_vision_rag.ipynb  # ColPali 비전 검색 성능 최적화
│   ├── 03_method2_dual_path.ipynb   # 듀얼 패스 시스템 통합 테스트
│   └── 04_evaluation_results.ipynb # [Main] 6개 메트릭 기반 통합 성능 리포트
│
├── data/                           # 🗂️ 데이터 저장소
│   ├── raw/                        # 원본 PDF 파일 보관
│   ├── chroma_db/                  # 텍스트 벡터 인덱스 저장소
│   └── .byaldi/                    # 비전 모델 전용 인덱스 저장소
│
├── config/                         # ⚙️ 프로젝트 세부 설정 (config.yaml)
├── requirements.txt                # 패키지 의존성 정의
├── project_structure.txt           # 파일별 상세 역할 가이드 (자동 생성)
└── README.md                       # 통합 가이드 (본 문서)
```
---
*(본 프로젝트는 금융 데이터 분석의 정교함을 높이기 위한 멀티모달 기술의 실질적인 적용 사례를 제시합니다.)*