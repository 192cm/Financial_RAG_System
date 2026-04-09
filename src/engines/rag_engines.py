from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from src.models import get_gemini_model
from src.retrieval.router import extract_target_company, extract_target_table, rerank_by_table, get_company_filtered_retriever
from src.retrieval.reranker import NeuralReranker
from src.engines.agent import CorrectiveAgent
from src.utils.vision import get_page_text_and_image
from src.utils.common import clean_llm_response

class RAGEngine:
    """Method 0-3의 모든 RAG 로직을 통합 관리하는 엔진"""
    
    def __init__(self, base_retriever, all_docs, vision_retriever=None):
        self.base_retriever = base_retriever
        self.all_docs = all_docs
        self.vision_retriever = vision_retriever
        self.agent = CorrectiveAgent()
        self.llm = get_gemini_model()
        self.reranker = NeuralReranker() # BGE-Reranker-v2-m3 로컬 로드
        self.retriever_cache = {} # 기업별 검색기 캐시 추가

    def run_method0_baseline(self, query: str, return_metadata: bool = False) -> Any:
        """Method 0: 단순 텍스트 기반 하이브리드 Baseline RAG를 실행합니다."""
        prompt = ChatPromptTemplate.from_template("다음 문서를 바탕으로 질문에 답하세요.\n\n문서:{context}\n질문:{input}")
        chain = create_retrieval_chain(self.base_retriever, create_stuff_documents_chain(self.llm, prompt))
        res = chain.invoke({"input": query})
        
        answer = res["answer"]
        # LangChain의 response에서 metadata 추출 (모델에 따라 위치가 다를 수 있음)
        # sota_engine.llm 호출이 아닌 chain 호출이므로, 내부 llm의 response를 확인해야 함
        # 여기선 호출 후의 통계 정보를 시뮬레이션하거나 수집 로직 추가
        metadata = {"usage": getattr(res.get("context", [None])[0], "usage_metadata", {})} # 예시
        
        return {"answer": answer, "metadata": metadata} if return_metadata else answer

    def run_method1_vision(self, query: str, return_metadata: bool = False) -> Any:
        """Method 1: ColPali 모델을 이용해 비전 정보로만 답변을 생성합니다."""
        if not self.vision_retriever: return "Error: Vision retriever not initialized"
        results = self.vision_retriever.search(query, k=1)
        b64 = results[0].base64
        msg = HumanMessage(content=[
            {"type": "text", "text": f"이미지 내용을 분석하여 질문에 답하세요.\n\n질문: {query}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ])
        res = self.llm.invoke([msg])
        answer = clean_llm_response(res.content)
        metadata = {"usage": getattr(res, "usage_metadata", {})}
        
        return {"answer": answer, "metadata": metadata} if return_metadata else answer

    def run_method2_dual_basic(self, query: str, return_metadata: bool = False) -> Any:
        """Method 2: 텍스트 검색과 비전 검색 결과를 단순 결합하여 답변을 생성합니다."""
        docs = self.base_retriever.invoke(query)
        context = "\n".join([d.page_content for d in docs[:3]])
        _, img = get_page_text_and_image(docs[0].metadata['source'], docs[0].metadata['page'], self.all_docs)
        
        content = [{"type": "text", "text": f"텍스트와 이미지를 참고하여 답변하세요.\n\n[텍스트]\n{context}\n\n[질문]\n{query}"}]
        if img: content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
        
        res = self.llm.invoke([HumanMessage(content=content)])
        answer = clean_llm_response(res.content)
        metadata = {"usage": getattr(res, "usage_metadata", {})}
        
        return {"answer": answer, "metadata": metadata} if return_metadata else answer

    def run_method3_sota(self, query: str, use_prefilter: bool = True, use_reranker: bool = True, max_expansions: int = 3, return_metadata: bool = False) -> Any:
        """Method 3 (SOTA): 라우팅, Dual-Path 검색, 그리고 에이전트 자가 교정 피드백 루프를 적용하여 답변을 도출합니다."""
        # 1. 대상 기업 추출 및 필터링된 검색기 결정 (Phase 1)
        company = "ALL"
        active_retriever = self.base_retriever
        if use_prefilter:
            company = extract_target_company(query)
            if company in self.retriever_cache:
                active_retriever = self.retriever_cache[company]
            else:
                active_retriever = get_company_filtered_retriever(self.base_retriever, self.all_docs, company)
                self.retriever_cache[company] = active_retriever

        # 2. Dual-Path Hybrid Retrieval (Phase 2 - Retrieval)
        # Path A: 텍스트 기반 검색
        text_retrieved = active_retriever.invoke(query)
        
        # Path B: 비전 기반 검색 (ColPali)
        vision_retrieved = []
        if self.vision_retriever:
            try:
                # 레이턴시와 정확도 균형을 위해 k=5 설정
                v_results = self.vision_retriever.search(query, k=5)
                # 비전 검색 결과는 전역 검색이므로, 추출된 기업명으로 필터링
                for res in v_results:
                    # [버그 수정] doc_id가 int로 반환되는 경우 대응
                    doc_path = getattr(res, 'doc_id', "")
                    if isinstance(doc_path, int):
                        # metadata가 있다면 거기서 실제 경로를 가져오고, 없으면 문자열 변환
                        doc_path = getattr(res, 'metadata', {}).get('doc_id', str(doc_path))
                    
                    # 기업명 포함 여부 확인 (필터링)
                    if company == "ALL" or (company.lower() in str(doc_path).lower()):
                        # Document 객체로 변환하여 병합 준비
                        p_num = res.page_num - 1 if hasattr(res, 'page_num') else 0
                        
                        # doc_path를 문자열로 비교하여 매칭 정확도 향상
                        target_doc = next((d for d in self.all_docs if str(d.metadata.get('source')) == str(doc_path) and d.metadata.get('page') == p_num), None)
                        if target_doc:
                            vision_retrieved.append(target_doc)
            except Exception as e:
                print(f"⚠️ 비전 검색 중 오류 발생 (무시하고 텍스트 검색 결과만 사용): {e}")

        # 3. Aggregator: 결과 통합 및 중복 제거 (+ 상호 검증 가중치 부여)
        text_keys = {(d.metadata.get('source'), d.metadata.get('page')) for d in text_retrieved}
        vision_keys = {(d.metadata.get('source'), d.metadata.get('page')) for d in vision_retrieved}
        
        combined = text_retrieved + vision_retrieved
        seen_keys = set()
        retrieved = []
        for doc in combined:
            src = doc.metadata.get('source', '')
            pg = doc.metadata.get('page', -1)
            key = (src, pg)
            if key not in seen_keys:
                seen_keys.add(key)
                # [고도화] 두 경로 모두에서 발견된 페이지에 상호 검증 보너스(2.0) 부여
                doc.metadata["consensus_boost"] = 2.0 if key in text_keys and key in vision_keys else 0.0
                retrieved.append(doc)

        # 4. Two-stage Reranking (Phase 2 - Reranking)
        # Step 1. Neural Reranking (Semantic Match)
        if use_reranker and self.reranker:
            retrieved = self.reranker.rerank(query, retrieved, top_k=len(retrieved))

        # Step 2. Heuristic Reranking (Table Name & Consensus Boost)
        table_name = extract_target_table(query)
        for doc in retrieved:
            # 기본 Neural Score (0~1)
            final_score = doc.metadata.get("rerank_score", 0)
            
            # [고도화] 상호 검증 보너스 합산
            final_score += doc.metadata.get("consensus_boost", 0.0)
            
            # 표 이름 가산점 합산
            if table_name != "NONE" and table_name in doc.page_content:
                final_score += 10.0
            
            doc.metadata["hybrid_score"] = final_score
        
        # 통합 점수 기준으로 최종 재정렬
        retrieved = sorted(retrieved, key=lambda x: x.metadata.get("hybrid_score", 0), reverse=True)

        # 5. Agentic Visual Reasoning (Phase 3)
        # retrieval 순서를 유지하면서 연속 페이지를 그룹으로 통합
        page_lookup = {
            (d.metadata.get('source', ''), d.metadata.get('page', -1)): d
            for d in retrieved
        }
        consumed = set()
        groups = []
        for doc in retrieved:
            source = doc.metadata.get('source', '')
            page = doc.metadata.get('page', -1)
            key = (source, page)

            if key in consumed: continue

            # 이 페이지 기준으로 앞뒤 연속 페이지 수집 (연속된 표 탐색 최적화)
            group_pages = [page]
            next_p = page + 1
            while (source, next_p) in page_lookup and (source, next_p) not in consumed:
                group_pages.append(next_p)
                next_p += 1
            prev_p = page - 1
            while (source, prev_p) in page_lookup and (source, prev_p) not in consumed:
                group_pages.insert(0, prev_p)
                prev_p -= 1

            for p in group_pages: consumed.add((source, p))
            groups.append({'source': source, 'pages': sorted(group_pages)})

        total_metadata = {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

        for group in groups:
            # 에이전트 실행 (자가 교정 루프)
            res_data = self.agent.run(query, group['source'], group['pages'], self.all_docs, max_expansions, table_name, return_metadata=True)
            
            # 메타데이터 누적
            agent_usage = res_data.get("metadata", {}).get("usage", {})
            for k in total_metadata["usage"]:
                if k in agent_usage:
                    total_metadata["usage"][k] += agent_usage[k]

            res = res_data["answer"]
            
            # [고도화] 정답을 찾았더라도 '확인 불가' 등 부정적인 답변인 경우 다음 후보군을 더 탐색하도록 개선
            negative_keywords = ["확인 불가", "기재되어 있지 않", "찾을 수 없", "명시되어 있지 않", "불가능"]
            is_negative = any(kw in res for kw in negative_keywords)
            
            # 정답을 확실히 찾았고 부정적인 답변이 아닌 경우에만 즉시 반환
            if res not in ["WRONG_DOCUMENT", "NOT_FOUND_IN_THIS_CANDIDATE"] and not is_negative:
                return {"answer": res, "metadata": total_metadata} if return_metadata else res
            
            # 만약 모든 후보를 뒤졌는데도 부정적인 답변만 있다면, 마지막 부정 답변이라도 반환하기 위해 저장
            last_res = res
        
        return {"answer": last_res if 'last_res' in locals() else "정답 찾기 실패", "metadata": total_metadata} if return_metadata else (last_res if 'last_res' in locals() else "정답 찾기 실패")
