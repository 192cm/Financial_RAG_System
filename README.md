# 📈 지능형 멀티모달 금융 RAG 시스템
### Intelligent Multi-modal Financial RAG with Vision Self-Correction Agent

본 프로젝트는 금융감독원 전자공시시스템(DART)의 방대한 비정형 멀티모달 사업보고서(PDF)에서 데이터를 정밀하게 추출하기 위해 설계된 **전문가용 하이브리드 RAG 시스템**입니다. 텍스트와 시각 정보를 동시에 처리하는 **Dual-Path Retrieval**과 에이전트 기반의 **자가 교정(Corrective Agent)** 아키텍처를 통해 데이터 왜곡 없는 고성능 금융 분석 환경을 제공합니다.

---

## 🚀 주요 혁신 기술 (Key Innovations)

### 1. Vision-First Retrieval (ColPali v1.2)
금융 보고서의 핵심인 표, 차트, 복잡한 레이아웃을 텍스트 추출 없이 이미지 본연의 의미로 검색합니다. 시각적 맥락을 유지하여 텍스트 기반 검색의 한계를 극복했습니다.

### 2. Dual-Path Hybrid Architecture
- **Text Path**: BM25 + Semantic Search (Ensemble)
- **Vision Path**: ColPali-based Vision Indexing
- **Consensus Boost**: 두 경로가 공통으로 지목한 페이지에 가중치를 부여하여 검색 신뢰도를 비약적으로 향상시켰습니다.

### 3. Neural Reranking (BGE-M3)
통합된 모든 검색 후보군에 대해 로컬 GPU를 활용한 **BAAI/bge-reranker-v2-m3** 기반 재순위화를 수행하여 검색 정밀도(Precision)를 극대화합니다.

### 4. Agentic Self-Correction (Corrective Agent)
Gemini 모델이 검색된 고화질 이미지를 직접 비전 분석하며 다음과 같은 지능형 작업을 수행합니다.
- **표 단절 감지**: 페이지 경계에서 표가 끊긴 경우 전후 페이지를 동적으로 탐색하는 **Sliding Window** 작동.
- **수치 검증**: 질문과 상관없는 데이터를 걸러내고 단위 누락을 자가 교정.
- **무한 루프 방지**: 에이전트의 최대 탐색 횟수를 제어하여 시스템 안정성을 확보했습니다.

---

## 📊 벤치마크 결과 (Comprehensive Benchmark)

10개의 고난도 금융 도메인 질문 세트(수치 계산, 표 분석 포함)에 대한 방법론별 성능 비교입니다.

| 모델 (Methodology) | Exact Match | ROUGE-L | BLEU | Latency (sec) | 특징 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Method 0 (Baseline)** | 0.6 | 0.380 | 0.022 | 10.04 | 텍스트 하이브리드 베이스라인 |
| **Method 1 (Vision Only)** | 0.0 | 0.249 | 0.006 | 12.49 | ColPali 단독 검색 |
| **Method 2 (Dual Basic)** | 0.3 | 0.289 | 0.026 | 9.97 | 단순 시각 정보 결합 |
| **w/o Reranker** | 0.7 | 0.389 | 0.070 | 41.20 | Reranking 제외 버전 |
| **SOTA (Full)** | **0.8** | **0.438** | **0.077** | **133.75** | **전체 아키텍처 및 에이전트 적용** |

> [!IMPORTANT]
> **SOTA 모델**은 에이전트의 자가 교정 및 다단계 재순위화 프로세스로 인해 Latency가 높으나, **Exact Match 80%** 및 **LLM Judge Score 4.2/5.0**으로 가장 높은 정답률과 답변 퀄리티를 보장합니다.

---

## 🛠️ 시작하기 (Getting Started)

### 1. 환경 준비 (Prerequisites)
- **OS**: Windows (tested) / Linux (Cuda support recommended)
- **Python**: 3.10+
- **Hardware**: GPU (8GB+ VRAM recommended for ColPali & Reranker)

### 2. 설치
```bash
git clone https://github.com/your-repo/Financial_RAG_System.git
cd Financial_RAG_System
pip install -r requirements.txt
```

### 3. 환경 설정
`.env` 파일을 루트 디렉토리에 생성하고 Google API 키를 입력합니다.
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. 하드웨어 체크 및 실행
```bash
# 하드웨어 사양 및 CUDA 가속 확인
jupyter notebook notebooks/00_environment_check.ipynb
```

---

## 📂 프로젝트 구조 (Project Structure)

```text
Financial_RAG_System/
├── src/                        # 🛠️ 핵심 엔진 모듈
│   ├── retrieval/              # 검색 레이어 (Vision, Text, Reranker, Router)
│   ├── engines/                # 실행 엔진 (RAG Workflow, Corrective Agent)
│   ├── evaluation/             # 평가 라이브러리 (Judge, Metrics, Runner)
│   └── utils/                  # 멀티모달 전처리 및 공통 유틸리티
├── notebooks/                  # 📓 방법론별 실험 및 결과 분석
│   ├── 00_environment_check.ipynb
│   ├── 01_baseline_rag.ipynb   # Method 0
│   ├── 02_method1_vision_rag.ipynb # Method 1
│   ├── 03_method2_dual_path.ipynb  # Method 2
│   ├── 04_method3_sota_rag.ipynb   # SOTA (Method 3)
│   └── 05_evaluation_results.ipynb # 최종 벤치마크 분석
├── data/                       # 🗂️ 원본 PDF 및 Vector DB / Index
└── config/                     # ⚙️ 시스템 설정 관리
```

---

## 🏗️ 시스템 아키텍처 (System Architecture)

### 상세 프로세스 설명

본 시스템은 금융 도메인의 특수성(정교한 표 데이터, 문서 레이아웃의 중요성)을 해결하기 위해 **Retriever-Reranker-Agent**의 3단계 파이프라인으로 구성되어 있습니다.

1.  **지능형 쿼리 라우팅 (Query Routing)**: 사용자의 질문에서 기업명, 보고서 종류, 특정 재무 항목을 자동 추출합니다. 이를 통해 검색 범위를 해당 기업의 문서로 즉각 좁혀 검색 노이즈를 원천 차단합니다.
2.  **Dual-Path Hybrid Retrieval**: 
    - **통합 텍스트 경로**: BM25(키워드 검색)와 Semantic Search(의미 기반 검색)의 앙상블을 통해 텍스트 중심의 정보를 포착합니다.
    - **멀티모달 비전 경로 (ColPali)**: 텍스트 추출 엔진(OCR 등)의 한계를 넘어, 문서 페이지 자체를 시각적 토큰으로 처리하여 검색합니다. 특히 복잡한 결산표나 차트를 검색하는 데 탁월한 성능을 발휘합니다.
3.  **상호 검증 기반 가중치 (Consensus Boost)**: 텍스트 경로와 비전 경로가 동시에 동일한 페이지를 지목할 경우, 해당 페이지가 정답일 확률이 매우 높다고 판단하여 높은 신뢰 점수를 부여합니다. 이는 개별 검색기의 단점을 상호 보완하는 핵심 로직입니다.
4.  **Neural Reranking**: 통합된 상위 후보군에 대해 고성능 **BGE-M3 Reranker**를 사용하여 질문과의 의미론적 유사도를 2차 정밀 계산합니다. 단순히 키워드가 겹치는 페이지가 아닌, 실제 질문의 의도에 부합하는 페이지를 최상단으로 정렬합니다.
5.  **에이전틱 자가 교정 (Corrective Agent & Sliding Window)**:
    - 추출된 최상위 페이지의 **고해상도 이미지**를 Gemini 비전 모델이 직접 분석하여 답변 가능 여부를 판단합니다.
    - 만약 재무제표의 표가 페이지 경계에서 끊겨 정보가 부족하다고 판단되면, 에이전트가 스스로 **Sliding Window** 메커니즘을 가동하여 전후 페이지(N-1, N+1)를 추가로 탐색합니다.
    - 수치 누락이나 단위 오인 가능성을 감지하면 스스로 다시 검색을 요청하거나 교정하여, 재무 데이터 추출 시 발생할 수 있는 '환각(Hallucination)' 현상을 전문 금융가 수준으로 통제합니다.

---

## 📖 링크

* Link: https://kunho192.tistory.com/16
