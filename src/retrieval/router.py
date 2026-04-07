import re
from src.models import get_gemini_model

def extract_target_company(query: str) -> str:
    """질문에서 분석 대상 기업명을 추출합니다."""
    llm = get_gemini_model(temperature=0)
    
    prompt = f"""사용자의 질문에서 '분석 대상 기업명'만 정확히 추출하세요. 
대상 기업이 명확하지 않거나 여러 기업인 경우 'ALL'이라고 답하세요.
복잡한 설명 없이 단어 하나만 출력하세요.

[질문]
{query}

결과:"""
    
    res = llm.invoke(prompt).content
    if isinstance(res, list):
        res = res[0].get("text", "") if isinstance(res[0], dict) else str(res[0])
    return res.strip()

def get_company_filtered_retriever(base_retriever, all_docs, company_name: str, k: int = 10):
    """특정 기업에 대해 필터링된 하이브리드 검색기를 생성합니다."""
    from langchain_classic.retrievers import EnsembleRetriever, BM25Retriever
    
    if company_name == "ALL":
        return base_retriever
        
    company_docs = [doc for doc in all_docs if company_name in doc.metadata.get('source', '')]
    
    if not company_docs:
        return base_retriever
        
    bm25 = BM25Retriever.from_documents(company_docs)
    bm25.k = k
    
    parent = base_retriever.retrievers[1]
    parent.search_kwargs["filter"] = {"source": {"$contains": company_name}}
    
    return EnsembleRetriever(retrievers=[bm25, parent], weights=[0.5, 0.5])
