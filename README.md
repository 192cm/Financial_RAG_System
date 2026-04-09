# 📈 지능형 멀티모달 금융 RAG 시스템 
### (Intelligent Multi-modal Financial RAG with Vision Self-Correction Agent)

본 프로젝트는 금융감독원 전자공시시스템(DART)의 방대한 비정형/멀티모달 사업보고서(PDF)에서 사용자가 원하는 재무 데이터를 왜곡 없이 정밀하게 추출하기 위해 설계된 **전문가용 하이브리드 RAG 시스템**입니다.

---

## ✨ 핵심 기술 및 특징 (Core Technologies)

1. **ColPali v1.2 기반 비전 검색**: 텍스트 추출의 한계를 넘어 이미지(표, 차트) 자체를 벡터화하여 검색하는 최신 시각적 리트리벌(Vision Retrieval) 방식 도입.
2. **로컬 Neural Reranking (BGE-M3)**: 검색된 후보군에 대해 BGE-Reranker-v2-m3 모델을 활용하여 문맥적 유사도를 재순위화함으로써 검색 정밀도 극대화.
3. **양방향 자가 교정(Bidirectional Corrective) 에이전트**: Gemini 모델이 검색된 이미지를 직접 읽고 질문과의 정합성을 스스로 검증하며, 표 단절이나 단위(Unit) 누락 감지 시 전후 페이지를 동적으로 탐색하는 **양방향 슬라이딩 윈도우(Bidirectional Sliding Window)** 루프 구현.
4. **하이브리드 앙상블 검색 & 재순위화**: BM25, Vector Semantics, ColPali 비전 검색을 병행하고, 질문 내 '표 이름' 기반 가중치와 Neural Score를 결합한 **Hybrid Reranking** 전략 사용.
5. **Context Carry-over 지능**: 여러 페이지에 걸친 분석 중 발견된 핵심 메타데이터(단위, 기업명 등)를 휘발시키지 않고 다음 탐색 단계로 전달하여 일관된 답변 생성 보장.

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

- **Method 0 (Baseline)**: 기존의 텍스트 기반 하이브리드(BM25 + Semantic) 검색 및 생성.
- **Method 1 (Vision-only)**: ColPali를 이용해 PDF 페이지 자체를 검색하여 멀티모달 LLM으로 답변 생성.
- **Method 2 (Dual-Path)**: 텍스트 검색 결과와 비전 검색 결과를 결합하여 최적의 컨텍스트 제공.
- **Method 3 (SOTA - Agentic Multimodal)**: 지능형 라우팅, **Hybrid Reranking(표 매칭 + Neural)**, 그리고 비전 기반 자가 교정 메커니즘을 결합. 에이전트가 시각적 단절을 감지하면 **전후 페이지로 검색 범위를 동적으로 확장(Bidirectional Sliding Window)**하고, 추출된 문맥(Context Carry-over)을 유지하며 최종 답변 도출.

---
*(본 프로젝트는 금융 데이터 분석의 정교함을 높이기 위한 멀티모달 기술의 실전 적용 사례를 제시합니다.)*