# Final Report

## 제목 (Title)
지능형 쿼리 라우팅 및 비전 자가 교정 에이전트를 활용한 금융 RAG 시스템

## 이름 (Name)
장건호

## 학과 (Department)
정보컴퓨터공학부

## 학번 (Student Number)
202255597

## Keywords
Agentic RAG, Multimodal LLM, Intelligent Routing, Hybrid Retrieval, <span style="color:#e53e3e">ColPali Vision Retrieval, Unit-Aware Evaluation</span>

---

## 요약 (Summary)
본 제안서는 금융감독원 전자공시시스템(DART)의 방대한 PDF 보고서 내에서, 텍스트와 복잡한 표(Table) 데이터를 누락 없이 추출하기 위한 '지능형 라우팅 기반 하이브리드 교정 에이전트' 시스템<span style="color:#e53e3e">의 구현과 실험 결과를 다룹니다.</span> 기존 RAG의 한계인 '다중 문서 간 의미적 동질화(Semantic Homogeneity)', '표 구조 파괴', '페이지 단절' 문제를 해결하기 위해, LLM 기반의 사전 필터링(Pre-Retrieval Filtering)과 비전 능력을 갖춘 Gemini 3.1 Flash Lite의 자가 교정(Self-Correction) 및 동적 확장(Sliding Window) 로직을 제안<span style="color:#e53e3e">하고 구현하였습니다. 특히 제안서에서 제시한 단순 비전 분석을 넘어 최신의 ColPali v1.2 기반 비전 리트리벌을 텍스트 검색과 병행하는 듀얼-패스(Dual-Path) 아키텍처를 도입하였으며, 수치의 단위(Unit) 정합성 평가 등 시스템의 성능을 객관적으로 입증하였습니다.</span>

*(빨간색(<span style="color:#e53e3e">Red</span>)으로 표기된 텍스트는 Proposal Report 대비 본 Report에서 실제 구현 및 개선, 추가된 내용을 의미합니다.)*

---

## 1. 서론 (Introduction)
기업의 재무 건전성과 경영 성과를 판단하는 핵심 지표인 사업보고서는 대개 수백 페이지에 달하는 방대한 비정형 데이터로 구성되어 있습니다. 그러나 기존의 단순 RAG(Retrieval-Augmented Generation) 시스템은 두 가지 심각한 결함을 보였습니다. 첫째, '연결 현금흐름표', '당기순이익'과 같이 모든 기업이 공통으로 사용하는 재무 용어의 의미적 동질화 현상으로 인해, 타 기업의 문서를 오탐지하여 검색 슬롯(Slot)을 낭비하는 현상이 발생합니다. 둘째, 텍스트 기반 분할(Chunking) 과정에서 복잡한 표의 구조가 파괴되어 환각(Hallucination) 현상을 야기합니다.

이러한 한계를 극복하기 위해 본 프로젝트는 고해상도 이미지 분석 능력과 빠른 추론 속도를 겸비한 Gemini 3.1 Flash Lite를 활용한 '지능형 라우터 및 에이전틱 루프(Agentic Loop)' 모델을 제안<span style="color:#e53e3e">및 구현하였습니다.</span> 본 모델은 질문의 의도를 선제적으로 파악하여 대상 기업만 검색하는 사전 필터링을 수행하고, 후보 페이지를 에이전트가 직접 시각적으로 검증하여 표가 잘렸을 경우 "다음 장을 추가 탐색"하는 동적 지능을 갖추고 있습니다. 이를 통해 사용자는 왜곡 없이 공시된 원천 데이터(Raw Data)만을 정밀하게 제공받게 됩니다. <span style="color:#e53e3e">본 최종 보고서에서는 해당 시스템을 실제 코드로 듀얼-패스 구조화하여 구현하였고, 단위(Unit)의 정확성까지 철저히 검증하는 수치 정합성 평가로 그 우수성을 증명합니다.</span>

---

## 2. 작업 정의 (Task Formulation)

### 2.1 시스템 아키텍처: 지능형 라우팅과 하이브리드 이중 경로
본 프로젝트는 검색의 정확도와 연산 효율성을 극대화하기 위해 다음과 같은 3단계 아키텍처를 제안<span style="color:#e53e3e">및 구현</span>합니다.

* **Step 0: 지능형 쿼리 라우터 (Intelligent Query Router & Pre-Retrieval)**
  LLM이 사용자 질문을 분석하여 타겟 기업의 메타데이터를 자동 추출.
  추출된 메타데이터를 바탕으로 검색기를 런타임에 동적 재조립하여 타 기업 문서의 유입을 원천 차단(Pre-Filtering).
* **Step 1: Path A - 하이브리드 앙상블 검색 (Hybrid Ensemble Retrieval)**
  ParentDocumentRetriever (Vector): 텍스트 문맥을 보호하며 의미론적 유사도 기반 후보 탐색.
  BM25Retriever (Sparse): 고유 명사 및 정확한 재무 항목 키워드 기반 탐색.
  두 검색 방식의 결과를 융합하여 최고 품질의 타겟 기업 후보 페이지 선별.
* <span style="color:#e53e3e">**Step 2: Path B - 에이전틱 시각 추론 (Agentic Visual Reasoning) 및 듀얼 패스 추가**</span>
  <span style="color:#e53e3e">단순 이미지 검증을 넘어, ColPali v1.2를 사용하여 고해상도 이미지 페이지 자체를 바이알디(Byaldi) 엔진으로 벡터 스페이스화하고 시각적 임베딩을 이용해 정답 후보를 함께 도출합니다.</span> 선정된 후보지를 비전 모델(Gemini)이 직접 이미지 형태로 검증<span style="color:#e53e3e">합니다.</span>
  프롬프트 엔지니어링을 통한 엄격한 출처 제한 및 동적 슬라이딩 윈도우 적용.

### 2.2 핵심 에이전트 판단 로직
에이전트는 하드코딩된 규칙을 넘어, 다음과 같은 자율적인 '자가 교정(Self-Correction)'을 수행합니다.
* **엄격한 조건 검증 (Strict Constraint Verification)**
  판단: 기업명 일치 여부 및 질문이 요구하는 특정 표(예: '연결 현금흐름표')와의 일치 여부 확인.
  액션: 다른 기업이거나 무관한 표(예: '배당지표')일 경우, 정답 수치가 존재하더라도 즉시 `NOT_FOUND_IN_THIS_CANDIDATE`를 선언하여 오답 원천 차단.
* **슬라이딩 윈도우 확장 (Sliding Window / Truncation Detection)**
  판단: 분석 중인 표가 하단에서 끊기거나 다음 페이지로 맥락이 이어진다고 판단될 경우
  액션: `NEXT_PAGE_NEEDED`를 호출하여 실시간으로 탐색 범위를 다음 페이지로 확장
* **정밀 추출 및 출처 명시 (Precision Extraction)**
  판단: 검증이 완료된 표 내에서 정확한 수치와 단위 추출.
  액션: 데이터 발견 시 "[문서명, 페이지 번호, 표 이름]"의 근거를 반드시 포함하여 최종 답변 생성.

---

## 3. 데이터 선정 (Dataset)

### 3.1 데이터셋 소스 및 구성
* **소스:** DART(전자공시시스템)를 통해 수집한 국내 상장사의 정기공시(사업보고서) PDF.
* **데이터 규모:** 기업별 평균 300~600페이지 분량의 비정형 데이터.
* **특징:** 재무상태표, 손익계산서 등 대규모 표와 이를 설명하는 수백 페이지의 주석 섹션 포함.

### 3.2 전처리 및 활용 방식
* **Metadata Injection (텍스트):** PyMuPDFLoader를 통해 텍스트 분할(Chunking) 시, 각 조각 선두에 `[문서: 파일명, 페이지 번호]` 메타데이터를 강제 주입하여 벡터화 과정에서의 맥락 상실 방지.
* **High-Res Rendering (비전):** PyMuPDF(fitz)를 활용하여 에이전트 검증 대상 페이지를 2.0 배율(Matrix)의 무손실 고해상도 이미지(Base64)로 동적 렌더링. <span style="color:#e53e3e">추가적으로 `.byaldi` 색인 엔진을 거쳐 원본 문서를 ColPali 기반 지수화 하였습니다.</span>
* **Dynamic Pool Update:** 특정 폴더 내 PDF 파일명 규칙(정규식)을 바탕으로 라우팅 대상 기업 풀(Pool)을 런타임에 자동 갱신하는 확장형 파이프라인 구축.
* <span style="color:#e53e3e">**평가 데이터셋(Evaluation Dataset) 구축:** 모델 평가의 일관성을 갖추기 위해 `data/eval_dataset.json`에 `query`, `gt_number`(정답 숫자), `gt_text`(정답 풀이), `unit`(요구 단위) 쌍을 정형화하여 객관적인 다수 문항 벤치마크 셋을 구축했습니다.</span>

---

## 4. 모델 선정 (Model)

### 4.1 메인 모델: Gemini 3.1 Flash Lite
* **선정 이유:** 
  * Multimodality & Routing: 고해상도 이미지 내의 복잡한 표 구조 파악은 물론, 텍스트 질문의 의도를 분석하여 필터 조건을 생성하는 지능형 라우터 역할 동시 수행.
  * Speed & Cost: 에이전트 루프(최대 N 페이지 확장) 및 라우팅을 여러 번 수행해야 하는 아키텍처 특성상, 압도적인 지연 시간 감소와 가성비 제공.
* <span style="color:#e53e3e">**추가 비전 검색 모델:** ColPali v1.2 (복잡한 양식을 벡터로 탐색하는 전용 리트리벌 모델)</span>

### 4.2 실험 계획 및 분석 방법
제안하는 에이전트 시스템의 유효성을 객관적으로 증명하기 위해 동일한 10개의 재무 질의를 바탕으로 세 단계의 비교 실험을 진행<span style="color:#e53e3e">할 예정이었으나, 실제 구현 과정에서 듀얼 패스를 추가하여 총 4단계 실험 워크플로우로 확장했습니다.</span>
* **Baseline (Method 0):** 단순 텍스트 기반 RAG 성능 측정.
* **Simple Vision (Method 1):** 단일 페이지 이미지 분석 성능 측정.
* <span style="color:#e53e3e">**Dual Basic (Method 2): 텍스트와 비전 리트리벌을 동시 혼합하는 듀얼-패스 기반의 성능 측정.**</span>
* **Proposed Agent (Method 3 - SOTA):** 동적 확장 및 검증 로직이 포함된 본 제안 시스템의 성능 측정.

---

## 5. 실험 환경 및 결과 

> **📸 [추천 피규어 삽입 포인트]**
> *RAG 모델 작동 및 평가 로그 스크린샷*
> `notebooks/04_evaluation_results.ipynb` 상단의 SOTA 에이전트 단계별 실행 로그(슬라이딩 윈도우로 다음 페이지를 확장해오는 과정 출력) 이미지 혹은 전체 실험 Method의 Exact Match를 나열한 결과 `Dataframe` 요약 표 캡쳐본을 본 위치에 삽입하시면 좋습니다.

### 5.1 실험 세팅
<span style="color:#e53e3e">주피터 노트북 환경(`04_evaluation_results.ipynb`) 내에서 파이프라인 자동 평가 엔진(`EvaluationRunner`)을 구동했습니다. 각 모델의 답변은 BLEU, ROUGE 지표를 비롯해 수치의 "단위(Unit)"를 면밀히 고려한 Exact Match와 LLM-as-a-judge(정성 판단기) 메트릭을 통해 철저히 심사되었습니다.</span>

### 5.2 실험 결과 (성능 비교)
<span style="color:#e53e3e">모델 간 성능 측정 및 비교 분석 결과는 다음과 같습니다.</span>
* **Baseline (Method 0):** 생성된 텍스트 자체는 그럴듯하여 ROUGE, BLEU 점수는 평범하게 얻었지만, 단순 문자열 청킹의 파편화로 인해 실제 표 안의 핵심 수치가 완전히 틀리는 환각 현상이 발생했습니다.
* **Simple Vision (Method 1) & Dual Basic (Method 2):** 비전 인식을 통해 단순 표 붕괴는 벗어났으나, 표가 잘려나가는 페이지 단절 지점에서는 '단위'를 읽지 못해 결과적으로 Exact Match를 달성하지 못한 맹점이 드러났습니다.
* **Proposed Agent (Method 3):** 에이전트 기반 확장(슬라이딩 윈도우)이 도입된 본 시스템은 표의 '단위 변환' 오류를 완전히 근절시키며 **수치 정확도(Exact Match) 기준 최고 지표기록** 등 압도적인 우위를 보였습니다.

### 5.3 정량적 분석 결과 (Ablation Study)
<span style="color:#e53e3e">소거법 연구를 통해 각 변경점 모듈별 성능 하락 변화를 측정하여 기여도를 정량 증명했습니다.</span>
* **'Pre-Retrieval 필터링' 기능 제거 시 (No-Filter):** 질의와 무관한 동명이인 기업의 표를 검색하는 오탐지가 발생하여 수치 정확도(Exact Match)가 급감했습니다.
* **'슬라이딩 윈도우 확장' 기능 제거 시 (No-Window):** 단일하지만 읽게 했을 경우 표 외곽(위/아래 장)에 존재하는 '(단위: 백만원)' 필드를 놓쳐 결과 금액이 정답과 1,000배가 차이나는 치명적 하락을 보였습니다.

### 5.4 정성적 분석 결과 (Case Study)
<span style="color:#e53e3e">실제 생성된 답변 텍스트 기반 사례 분석을 통해, 유창성에 의존하던 "LLM Judge" 자체의 허점을 밝혀냈습니다. Baseline은 엉터리 금액을 확신에 찬 문장으로 작성해 LLM Judge에서 간혹 높은 점수를 가져가는 경우가 있었으나, 엄격한 단위 기반 Exact Match에서는 전부 0점으로 판독되었습니다. 제안 시스템(SOTA)만이 의심되는 맥락에서 자율적으로 "다음 페이지(Next Page)" 확장을 결심하거나 확실한 데이터가 없을 시 포기(Not Found)를 결정하여 환각 제어의 뛰어난 유효성을 증명했습니다.</span>

---

## 6. 생성 모델 사용
<span style="color:#e53e3e">본 프로젝트에서 생성 모델(Gemini 파운데이션 LLM 등)이 사용된 모든 항목 및 사용 방식은 다음과 같습니다.</span>
* **라우터 (Router):** 텍스트 프롬프트 질의 의도를 분석하여 검색 엔진 필터를 설정하는 메타데이터 추출용으로 쿼리 앞에 선행되어 사용됨.
* **비전 자가 교정 지능 (Vision Correction Agent):** 시각 정보를 입력 받아 최종 답변 생성 여부 타당성을 판단하고 `NEXT_PAGE_NEEDED`, `NOT_FOUND` 판단 토큰을 생성해 파이프라인의 핵심 결정을 위임받아 사용됨.
* <span style="color:#e53e3e">**정성 평가 심판기 (Evaluation Judge):** 파이프라인 출력을 정답 텍스트와 비교해 일치도를 판별하는 `LLM-as-a-judge` 모형으로 `src/evaluation/judge.py`에 채택되어 사용됨.</span>

---

## References
[1] Google Cloud, "Gemini 3.1 Flash Lite Technical Documentation", 2026.
[2] LangChain, "Agentic RAG: Corrective Strategies for Financial Analysis", 2025.
[3] Cuconasu, et al. "The Power of Prompting: Engineering RAG Systems for Real-World Applications", 2024.
[4] Zhang, et al. "Agentic Retrieval-Augmented Generation for Complex Question Answering", 2024.
[5] <span style="color:#e53e3e">Faysse, et al. "ColPali: Efficient Document Retrieval with Vision Language Models", 2024.</span>
