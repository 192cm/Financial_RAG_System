from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

from src.models import get_gemini_model
from src.retrieval.router import extract_target_company, get_company_filtered_retriever
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
        for doc in retrieved:
            res = self.agent.run(query, doc.metadata['source'], doc.metadata['page'], self.all_docs, max_expansions)
            if res not in ["WRONG_DOCUMENT", "NOT_FOUND_IN_THIS_CANDIDATE"]:
                return res
        return "정답 찾기 실패"
