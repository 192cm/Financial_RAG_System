import sys
import os
# 프로젝트 루트 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from src.retrieval.reranker import NeuralReranker

def test_neural_reranker():
    print("🔍 Neural Reranker 테스트를 시작합니다...")
    
    try:
        # 모델명은 기본값 사용 (BAAI/bge-reranker-v2-m3)
        reranker = NeuralReranker()
        
        query = "삼성전자의 2024년 영업이익은 얼마인가요?"
        
        # 테스트용 가짜 문서들
        documents = [
            Document(page_content="삼성전자의 2024년 연결기준 영업이익은 30조 원을 기록했습니다.", metadata={"source": "report_a.pdf", "page": 1}),
            Document(page_content="SK하이닉스의 2024년 영업이익 전망치는 약 15조 원입니다.", metadata={"source": "report_b.pdf", "page": 5}),
            Document(page_content="어제 먹은 점심 메뉴는 김치찌개였습니다. 가격은 9,000원이었습니다.", metadata={"source": "diary.txt", "page": 1}),
            Document(page_content="삼성전자는 세계적인 반도체 및 가공 가전 기업으로 본사는 수원에 있습니다.", metadata={"source": "info.pdf", "page": 10})
        ]
        
        print(f"\n[질문]: {query}")
        print("-" * 50)
        
        # 재순위화 수행
        results = reranker.rerank(query, documents, top_k=4)
        
        for i, doc in enumerate(results):
            score = doc.metadata.get("rerank_score", 0)
            print(f"순위 {i+1} (Score: {score:.4f}):")
            print(f"내용: {doc.page_content[:60]}...")
            print(f"출처: {doc.metadata['source']} (Page {doc.metadata['page']})")
            print("-" * 50)
            
        print("\n✅ 테스크 완료! 점수가 높을수록 질문과 관련성이 높은 문서입니다.")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("참고: 'sentence-transformers'와 'torch' 라이브러리가 설치되어 있어야 합니다.")

if __name__ == "__main__":
    test_neural_reranker()
