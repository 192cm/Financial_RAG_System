# 📈 지능형 멀티모달 금융 RAG 시스템 
### (Intelligent Multi-modal Financial RAG with Vision Self-Correction Agent)

본 프로젝트는 금융감독원 전자공시시스템(DART)의 방대한 비정형/멀티모달 사업보고서(PDF)에서 사용자가 원하는 재무 데이터를 왜곡 없이 정밀하게 추출하기 위해 설계된 **전문가용 하이브리드 RAG 시스템**입니다.

---

## ✨ 핵심 기술 및 특징 (Core Technologies)

1. **ColPali v1.2 기반 비전 검색**: 텍스트 추출의 한계를 넘어 이미지(표, 차트) 자체를 벡터화하여 검색하는 최신 시각적 리트리벌(Vision Retrieval) 방식 도입.
2. **자가 교정(Corrective) 에이전트**: Gemini 모델이 검색된 이미지를 직접 읽고 질문과의 정합성을 스스로 검증하며, 오답 시 후보군을 확장하거나 검색 전략을 수정하는 루프 구현.
3. **하이브리드 앙상블 검색**: 고유명사에 강한 BM25와 맥락에 강한 Vector Semantics를 결합하여 전문 금융 용어 검색의 정확도 극대화.
4. **수치 정합성 평가**: 단순 텍스트 비교를 넘어 단위(천원, 백만원 등)를 고려한 수치 데이터 검증 로직을 평가 파이프라인에 통합.

---

## 📂 프로젝트 구조 (Project Structure)

```text
Financial_RAG_System/
│
├── src/                            # 🛠️ 핵심 엔진 모듈
│   ├── retrieval/                  # 검색 엔진 레이어
│   │   ├── vision_engine.py        # ColPali 기반 비전 인덱싱 및 검색
│   │   ├── text_engine.py          # BM25 + Vector 하이브리드 검색
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
│   │   └── common.py               # 응답 정제 및 공통 유틸
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
- **Method 3 (SOTA - Agentic)**: 검색 결과의 정합성을 에이전트가 판단하고, 필요 시 검색 범위를 확장하여 최종 답변 도출.

---
*(본 프로젝트는 금융 데이터 분석의 정교함을 높이기 위한 멀티모달 기술의 실전 적용 사례를 제시합니다.)*