# 📈 지능형 멀티모달 금융 RAG 시스템 
### (Intelligent Multi-modal Financial RAG with Vision Self-Correction Agent)

본 프로젝트는 금융감독원 전자공시시스템(DART)의 방대한 비정형/멀티모달 사업보고서(PDF)에서 사용자가 원하는 재무 데이터를 왜곡 없이 정밀하게 추출하기 위해 설계된 **전문가용 하이브리드 RAG 시스템**입니다.

---

## ✨ 핵심 기술 (Core Technologies)

1. **ColPali v1.2 기반 비전 검색**: 텍스트 추출의 한계를 넘어 이미지(표, 차트) 자체를 벡터화하여 검색하는 최신 시각적 리트리벌(Vision Retrieval) 방식 도입.
2. **Dual-Path Hybrid Retrieval**: 텍스트 앙상블(BM25+Semantic)과 비전 리트리벌을 병렬로 수행하여 검색 후보군을 통합함으로써 검색 누락률 최소화.
3. **상호 검증 가중치 (Consensus Boost)**: 두 검색 경로가 동시에 지목한 페이지에 대해 AI가 높은 신뢰 점수를 부여하여 정답 페이지를 최상단으로 강제 정렬.
4. **로컬 Neural Reranking (BGE-M3)**: 통합된 모든 후보군에 대해 의미론적 유사도를 재순위화함으로써 검색 정밀도 극대화.
5. **Bidirectional Corrective 에이전트**: Gemini 모델이 검색된 고화질 이미지를 직접 읽고(2차 비전 검증), 표 단절이나 단위 누락 감지 시 전후 페이지를 동적으로 탐색하는 **Sliding Window** 메커니즘.
6. **지능형 라우팅 & 캐싱**: 질문에서 기업명과 표 이름을 자동 추출하여 검색 범위를 좁히고, 반복 쿼리 시 검색기 로딩 시간을 단축하는 최적화 적용.

---

## 📊 벤치마크 결과 (Benchmark Results)

금융 질문 10개 세트에 대한 각 방법론별 성능 지표입니다. (상세 결과: [05_evaluation_results.ipynb](notebooks/05_evaluation_results.ipynb))

| 모델 (Method) | Exact Match | ROUGE-L | BLEU | Latency (sec) | 특징 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Method 0 (Baseline)** | 0.6 | 0.288 | 0.054 | 10.04 | 텍스트 하이브리드 베이스라인 |
| **Method 1 (Vision Only)** | 0.4 | 0.214 | 0.038 | 12.49 | ColPali 단독 검색 |
| **Method 2 (Dual Basic)** | 0.7 | 0.391 | 0.069 | 9.97 | 검색 결과 단순 결합 |
| **SOTA (Full)** | **0.8** | **0.438** | **0.077** | **133.76** | **전체 아키텍처 및 에이전트 적용** |

> [!TIP]
> **SOTA 모델**은 에이전트의 자가 교정 루프가 포함되어 Latency는 증가하지만, 복잡한 표 데이터에 대해 **Exact Match 80%**라는 압도적인 정밀도를 보여줍니다.

---

## 📂 프로젝트 구조 (Project Structure)

```text
Financial_RAG_System/
│
├── src/                            # 🛠️ 핵심 엔진 모듈
│   ├── retrieval/                  # 검색 엔진 레이어 (Vision, Text, Reranker, Router)
│   ├── engines/                    # 실행 엔진 레이어 (RAG Workflow, Corrective Agent)
│   ├── evaluation/                 # 평가 자동화 (Runner, Judge, Metrics)
│   ├── utils/                      # 유틸리티 (Vision, Common Helper)
│   ├── config.py                   # 중앙 설정 관리
│   └── models.py                   # LLM 팩토리 (Gemini 최적화)
│
├── notebooks/                      # 📓 분석 및 실험
│   ├── 01_baseline_rag.ipynb       # [Method 0] 테스트
│   ├── 02_method1_vision_rag.ipynb  # [Method 1] 테스트
│   ├── 03_method2_dual_path.ipynb   # [Method 2] 테스트
│   ├── 04_method3_sota_rag.ipynb   # [Method 3] SOTA 아키텍처 데모
│   └── 05_evaluation_results.ipynb # [Main] 최종 벤치마크 결과 분석
│
├── data/                           # 🗂️ 데이터 저장소 (PDF, DB, Dataset)
├── config/                         # ⚙️ YAML 설정 파일
└── README.md                       # 통합 가이드
```

---

## 🚀 주요 워크플로우

- **Method 2 (Dual-Path Basic)**: 텍스트 검색 결과와 시각 정보를 단순 결합하여 제공하는 하이브리드 RAG. ([03_method2_dual_path.ipynb](notebooks/03_method2_dual_path.ipynb))
- **Method 3 (SOTA - Agentic Multimodal)**: 지능형 라우팅과 **Dual-Path 검색**, **Consensus Boost**, **Neural Reranking**, **Corrective Agent**가 결합된 최신 아키텍처. 에이전트가 단절 감지 시 전후 페이지를 동적으로 탐색하며 최종 답변을 도출합니다. ([04_method3_sota_rag.ipynb](notebooks/04_method3_sota_rag.ipynb))

---
*(본 프로젝트는 금융 데이터 분석의 정교함을 높이기 위한 멀티모달 기술의 실전 적용 사례를 제시합니다.)*