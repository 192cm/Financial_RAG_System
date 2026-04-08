from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from src.models import get_gemini_model
from src.retrieval.router import extract_target_company, extract_target_table, rerank_by_table, get_company_filtered_retriever
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

    def run_method0_baseline(self, query: str) -> str:
        """Method 0: 단순 텍스트 기반 Baseline RAG"""
        prompt = ChatPromptTemplate.from_template("다음 문서를 바탕으로 질문에 답하세요.\n\n문서:{context}\n질문:{input}")
        chain = create_retrieval_chain(self.base_retriever, create_stuff_documents_chain(self.llm, prompt))
        return chain.invoke({"input": query})["answer"]

    def run_method1_vision(self, query: str) -> str:
        """Method 1: Simple Vision RAG (ColPali k=1)"""
        if not self.vision_retriever: return "Error: Vision retriever not initialized"
        results = self.vision_retriever.search(query, k=1)
        b64 = results[0].base64
        msg = HumanMessage(content=[
            {"type": "text", "text": f"이미지 내용을 분석하여 질문에 답하세요.\n\n질문: {query}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ])
        res = self.llm.invoke([msg]).content
        return clean_llm_response(res)

    def run_method2_dual_basic(self, query: str) -> str:
        """Method 2: Dual-Path Basic Hybrid (No Agent Loop)"""
        docs = self.base_retriever.invoke(query)
        context = "\n".join([d.page_content for d in docs[:3]])
        _, img = get_page_text_and_image(docs[0].metadata['source'], docs[0].metadata['page'], self.all_docs)
        
        content = [{"type": "text", "text": f"텍스트와 이미지를 참고하여 답변하세요.\n\n[텍스트]\n{context}\n\n[질문]\n{query}"}]
        if img: content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
        res = self.llm.invoke([HumanMessage(content=content)]).content
        return clean_llm_response(res)

    def run_method3_sota(self, query: str, use_prefilter: bool = True, max_expansions: int = 3) -> str:
        """Method 3: Full Agentic Dual-Path System"""
        active_retriever = self.base_retriever
        if use_prefilter:
            company = extract_target_company(query)
            active_retriever = get_company_filtered_retriever(self.base_retriever, self.all_docs, company)

        retrieved = active_retriever.invoke(query)

        # 표 이름이 질문에 명시된 경우, 해당 표를 포함한 페이지를 우선 배치
        table_name = extract_target_table(query)
        if table_name != "NONE":
            retrieved = rerank_by_table(retrieved, table_name)

        # (source, page) 기준으로 중복된 문서 제거
        seen_keys = set()
        unique_docs = []
        for doc in retrieved:
            key = (doc.metadata.get('source', ''), doc.metadata.get('page', -1))
            if key not in seen_keys:
                seen_keys.add(key)
                unique_docs.append(doc)

        # retrieval 순서(relevance rank)를 유지하면서 연속 페이지를 그룹으로 통합
        # - 그룹 우선순위: 그룹 내 첫 번째로 retrieve된 페이지의 rank
        # - 그룹 내 페이지 순서: page 번호 오름차순 (agent가 순서대로 읽을 수 있도록)
        page_lookup = {
            (d.metadata.get('source', ''), d.metadata.get('page', -1)): d
            for d in unique_docs
        }
        consumed = set()  # 이미 그룹에 포함된 (source, page)

        groups = []
        for doc in unique_docs:
            source = doc.metadata.get('source', '')
            page = doc.metadata.get('page', -1)
            key = (source, page)

            if key in consumed:
                continue

            # 이 페이지 기준으로 앞뒤 연속 페이지 수집
            group_pages = [page]
            next_p = page + 1
            while (source, next_p) in page_lookup and (source, next_p) not in consumed:
                group_pages.append(next_p)
                next_p += 1
            prev_p = page - 1
            while (source, prev_p) in page_lookup and (source, prev_p) not in consumed:
                group_pages.insert(0, prev_p)
                prev_p -= 1

            for p in group_pages:
                consumed.add((source, p))

            groups.append({'source': source, 'pages': sorted(group_pages)})

        for group in groups:
            res = self.agent.run(query, group['source'], group['pages'], self.all_docs, max_expansions, table_name)
            if res not in ["WRONG_DOCUMENT", "NOT_FOUND_IN_THIS_CANDIDATE"]:
                return res
        return "정답 찾기 실패"
