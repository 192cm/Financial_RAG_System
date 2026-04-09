# 📈 지능형 멀티모달 금융 RAG 시스템 
### (Intelligent Multi-modal Financial RAG with Vision Self-Correction Agent)

본 프로젝트는 금융감독원 전자공시시스템(DART)의 방대한 비정형/멀티모달 사업보고서(PDF)에서 사용자가 원하는 재무 데이터를 왜곡 없이 정밀하게 추출하기 위해 설계된 **전문가용 하이브리드 RAG 시스템**입니다.

---

## ✨ 핵심 기술 및 특징 (Core Technologies)

1. **ColPali v1.2 기반 비전 검색**: 텍스트 추출의 한계를 넘어 이미지(표, 차트) 자체를 벡터화하여 검색하는 최신 시각적 리트리벌(Vision Retrieval) 방식 도입.
2. **Dual-Path Hybrid Retrieval**: 텍스트 앙상블(BM25+Semantic)과 비전 리트리벌을 병렬로 수행하여 검색 후보군을 통합함으로써 검색 누락률 최소화.
3. **상호 검증 가중치(Consensus Boost)**: 두 검색 경로가 동시에 지목한 페이지에 대해 AI가 높은 신뢰 점수를 부여하여 정답 페이지를 최상단으로 강제 정렬.
4. **로컬 Neural Reranking (BGE-M3)**: 통합된 모든 후보군에 대해 의미론적 유사도를 재순위화함으로써 검색 정밀도 극대화.
5. **양방향 자가 교정(Bidirectional Corrective) 에이전트**: Gemini 모델이 검색된 고화질 이미지를 직접 읽고(2차 비전 검증), 표 단절이나 단위 누락 감지 시 전후 페이지를 동적으로 탐색하는 **Sliding Window** 메커니즘.
6. **메모리 기반 검색기 캐싱(Retriever Caching)**: 동일 기업 반복 쿼리 시 검색기 필터링 비용을 0ms로 단축하는 최적화 적용.

---

## 📂 프로젝트 구조 (Project Structure)

```text
Financial_RAG_System/
│
├── src/                            # 🛠️ 핵심 엔진 모듈
│   ├── retrieval/                  # 검색 엔진 레이어
│   │   ├── vision_engine.py        # ColPali 기반 비전 인덱싱 및 검색
│   │   ├── text_engine.py          # BM25 + Vector 하이브리드 검색
│   │   ├── reranker.py             # BGE-Reranker-v2-m3 기반 재순위화
│   │   └── router.py               # 쿼리 기반 대상 기업 자동 라우팅
│   │
│   ├── engines/                    # 실행 엔진 레이어
│   │   ├── rag_engines.py          # Method 0-3 워크플로우 통합 관리
│   │   └── agent.py                # 가이딩된 자가 교정(Corrective) 에이전트
│   │
│   ├── evaluation/                 # 평가 자동화 레이어
│   │   ├── runner.py               # 벤치마크 실행 및 수치 정합성 검증
│   │   ├── judge.py                # LLM-as-a-Judge 기반 정성 평가
│   │   └── metrics.py              # EM, ROUGE-L, Numeric Accuracy 등 계산
│   │
│   ├── utils/                      # 공용 유틸리티
│   │   ├── vision.py               # 이미지/PDF 처리
│   │   ├── common.py               # 응답 정제 및 공통 유틸
│   │   └── test_reranker.py        # Reranker 로컬 작동 검증 스크립트
│   │
│   ├── config.py                   # 중앙 설정 관리
│   └── models.py                   # LLM 팩토리 (Gemini 최적화)
│
├── notebooks/                      # 📓 분석 및 실험
│   ├── 00_environment_check.ipynb  # 환경 점검
│   ├── 01_baseline_rag.ipynb       # [Method 0] 텍스트 하이브리드 RAG
│   ├── 02_method1_vision_rag.ipynb  # [Method 1] 비전 전용 RAG
│   ├── 03_method2_dual_path.ipynb   # [Method 2] 듀얼 패스 하이브리드 RAG
│   └── 04_evaluation_results.ipynb # [Main] 통합 벤치마크 및 결과 분석
│
├── data/                           # 🗂️ 데이터 저장소
│   ├── raw/                        # 원본 PDF 파일
│   ├── chroma_db/                  # 텍스트 벡터 인덱스
│   ├── .byaldi/                    # 비전 모델 전용 인덱스
│   ├── eval_dataset.json           # 정형화된 평가 셋
│   └── docstore.pkl                # 원본 문서 저장소
│
├── config/                         # ⚙️ 설정 파일 (config.yaml)
├── requirements.txt                # 의존성 정의
├── project_structure.txt           # 파일별 상세 역할 가이드
└── README.md                       # 통합 가이드
```

---

## 🚀 워크플로우 (Methodologies)

- **Method 2 (Dual-Path Basic)**: 텍스트 검색 결과와 시각 정보를 단순 결합하여 제공하는 하이 레벨 하이브리드 RAG. ([03_method2_dual_path.ipynb](file:///c:/Users/kyle0/Develops/Financial_RAG_System/notebooks/03_method2_dual_path.ipynb))
- **Method 3 (SOTA - Agentic Multimodal)**: 지능형 라우팅과 **Dual-Path 검색**, **Consensus Boost(상호 검증 가중치)**, **Neural + Heuristic 리랭킹**이 결합된 최신 아키텍처. 에이전트가 단절 감지 시 전후 페이지를 동적으로 탐색하며 최종 답변을 도출합니다. ([05_method3_sota_rag.ipynb](file:///c:/Users/kyle0/Develops/Financial_RAG_System/notebooks/05_method3_sota_rag.ipynb))

---
*(본 프로젝트는 금융 데이터 분석의 정교함을 높이기 위한 멀티모달 기술의 실전 적용 사례를 제시합니다.)*