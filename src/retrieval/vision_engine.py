import os
from byaldi import RAGMultiModalModel
from src.config import settings

class VisionRetrievalEngine:
    """ColPali 기반 비전 검색기 관리 클래스"""
    
    def __init__(self, model_name: str = "vidore/colpali-v1.2"):
        self.model_name = model_name
        self.retriever = None

    def load_index(self, index_name: str = None):
        """기존 인덱스를 로드합니다."""
        idx = index_name or settings.VISION_INDEX_NAME
        try:
            self.retriever = RAGMultiModalModel.from_index(idx)
            print(f"✅ 비전 인덱스 '{idx}' 로드 완료")
            return self.retriever
        except Exception as e:
            print(f"⚠️ 비전 인덱스 로드 실패: {e}")
            return None

    def index(self, input_path: str, index_name: str = None, overwrite: bool = True):
        """새로운 비전 인덱스를 생성합니다."""
        idx = index_name or settings.VISION_INDEX_NAME
        print(f"⏳ '{input_path}'의 PDF를 기반으로 새 비전 인덱스 '{idx}'를 구축 중... (시간이 소요될 수 있습니다)")
        self.retriever = RAGMultiModalModel.from_pretrained(self.model_name)
        self.retriever.index(
            input_path=input_path,
            index_name=idx,
            store_collection_with_index=True,
            overwrite=overwrite
        )
        print(f"✅ 비전 인덱스 '{idx}' 구축 완료!")
        return self.retriever

    def search(self, query: str, k: int = 1):
        """비전 검색을 수행합니다."""
        if not self.retriever:
            # 인덱스가 로드되지 않았다면 시도
            if not self.load_index():
                raise ValueError("검색기가 로드되지 않았고 로드 시도도 실패했습니다. index()를 통해 먼저 구축하세요.")
        return self.retriever.search(query, k=k)
