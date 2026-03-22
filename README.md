# 📈 지능형 쿼리 라우팅 및 비전 자가 교정 에이전트를 활용한 금융 RAG 시스템

본 프로젝트는 금융감독원 전자공시시스템(DART)의 방대한 비정형/멀티모달 사업보고서(PDF)에서 사용자가 원하는 재무 데이터를 왜곡 없이 정밀하게 추출하기 위한 **하이브리드 RAG 시스템**입니다.

기존 RAG 모델의 고질적인 문제인 **'다중 문서 간 의미적 동질화(Semantic Homogeneity)'**, **'표 구조 파괴'**, **'페이지 단절'** 문제를 지능형 라우팅과 멀티모달 자가 교정 에이전트를 통해 완벽하게 해결합니다.

---

## ✨ 핵심 기능 (Key Features)

1. **지능형 쿼리 라우터 (Intelligent Query Router)**
   - `data/raw/` 폴더 내의 PDF 파일명에서 기업명 리스트를 런타임에 동적으로 자동 추출합니다.
   - LLM(Gemini)이 사용자 질문의 의도를 분석하여 타겟 기업을 스스로 파악하고, 검색 엔진에 사전 필터(Pre-Retrieval Filter)를 씌워 타 기업 문서의 유입을 원천 차단합니다.

2. **하이브리드 앙상블 검색 (Hybrid Ensemble Retrieval)**
   - **Vector (ParentDocumentRetriever)**: 청크(Chunk)로 분할 시 메타데이터(`[문서: 파일명, 페이지 번호]`)를 강제 주입하고, 페이지 전체의 문맥을 보호하며 의미론적 검색을 수행합니다.
   - **Sparse (BM25Retriever)**: '연결포괄손익계산서' 등 정교한 금융 고유 명사를 키워드 기반으로 정확하게 탐색합니다.

3. **에이전틱 시각 검증 루프 (Agentic Visual Reasoning)**
   - 후보 페이지를 비전 모델(Gemini 3.1 Flash Lite)이 직접 2.0 배율 고해상도 이미지로 읽고 시각적으로 검증합니다.
   - **엄격한 조건 일치**: 사용자가 지목한 특정 표(예: '현금흐름표')가 아닐 경우, 정답 수치가 있더라도 `NOT_FOUND`로 쳐내는 강력한 제약을 갖추고 있습니다.
   - **슬라이딩 윈도우(Sliding Window)**: 표가 페이지 하단에서 잘렸다고 판단되면, 에이전트 스스로 `NEXT_PAGE_NEEDED`를 호출하여 다음 물리적 페이지로 탐색을 확장합니다.

---

## 🏗️ 시스템 아키텍처 (Architecture)

시스템은 다음 3단계로 동작합니다:

1. **Step 0 (Pre-Retrieval)**: 질문 입력 ➡️ LLM 라우터가 타겟 기업 추출 ➡️ 해당 기업 전용 필터 생성
2. **Step 1 (Path A)**: 하이브리드 검색기(BM25 + Vector)가 타겟 기업 문서 내에서 최적의 후보 페이지 묶음(Top-K) 선별
3. **Step 2 (Path B)**: 에이전트가 후보 페이지를 순회하며 이미지 기반으로 정답 여부를 판단 및 교정 ➡️ 최종 근거(출처)가 포함된 답변 도출

---

## 📂 디렉토리 구조 (Directory Structure)

```text
dual_path_vision_rag/
│
├── data/                           # 🗂️ 데이터 저장소
│   ├── raw/                        # 10개 기업 사업보고서 PDF
│   ├── chroma_db/                  # Vector DB
│   └── docstore.pkl                # ParentDocument 원본 텍스트 저장소
│
├── config/                         # ⚙️ 환경 설정 폴더
│   └── config.yaml                 # 설정값 (API Key, Chunk Size 등)
│
├── src/                            # 🛠️ [NEW!] 핵심 로직 모듈 (.py)
│   ├── __init__.py                 
│   ├── document_processor.py       # PDF 로드, 텍스트 Chunking (Parent Doc 세팅) 로직
│   ├── query_router.py             # 기업명 동적 인식 및 지능형 라우팅 로직
│   ├── vision_utils.py             # PDF를 고화질 이미지(Base64)로 변환하는 함수
│   └── agent_workflow.py           # N, N+1, N+2 동적 윈도우 & 교정적 에이전트 루프 클래스
│
├── notebooks/                      # 📓 실험 및 시각화 노트북 (.ipynb)
│   ├── 01_baseline_text_rag.ipynb      # src/ 모듈을 import 하여 대조군 실험
│   ├── 02_method1_vision_only.ipynb    # src/ 모듈을 import 하여 실험군 1 실험
│   ├── 03_method2_dual_path.ipynb      # src/ 모듈을 import 하여 최종 제안 모델 실행
│   └── 04_ablation_study_results.ipynb # 세 모델 결과 비교 막대그래프 시각화
│
├── .env                            # 보안 키
├── requirements.txt                # 패키지 의존성
└── README.md                       # 프로젝트 설명서
