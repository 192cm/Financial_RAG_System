import torch
from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class NeuralReranker:
    """BAAI/bge-reranker-v2-m3 모델을 이용한 로컬 기반 시맨틱 재순위화 클래스"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"🚀 Neural Reranker 로딩 중... (Device: {self.device})")
        # Sigmoid 활성화 함수를 기본으로 설정하여 점수를 [0, 1] 범위로 매핑
        self.model = CrossEncoder(model_name, device=self.device, default_activation_function=torch.nn.Sigmoid())
        print(f"✅ Reranker 모델 로드 완료: {model_name}")

    def rerank(self, query: str, documents: List[Document], top_k: int = 10) -> List[Document]:
        """질문과 문서 리스트를 받아 유사도 점수를 기반으로 재정렬합니다."""
        if not documents:
            return []

        # (질문, 문서내용) 쌍 구축
        pairs = [[query, doc.page_content] for doc in documents]
        
        # 점수 예측
        with torch.no_grad():
            scores = self.model.predict(pairs)
        
        # 문서 객체에 점수 메타데이터 추가
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)
            
        # 점수 기준 내림차순 정렬
        sorted_docs = sorted(documents, key=lambda x: x.metadata["rerank_score"], reverse=True)
        
        return sorted_docs[:top_k]
