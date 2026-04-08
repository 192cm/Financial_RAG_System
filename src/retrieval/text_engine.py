import os
import pickle
from typing import List, Tuple
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import (
    ParentDocumentRetriever,
    EnsembleRetriever,
    BM25Retriever,
)
from langchain_classic.storage import InMemoryStore
from src.config import settings

class TextRetrievalEngine:
    """텍스트 기반 하이브리드 검색기 관리 클래스"""
    
    def __init__(self, chunk_size: int = 400, overlap: int = 40):
        self.embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        
    def build_or_load(self, pdf_paths: List[str], force_rebuild: bool = False, k: int = 10) -> Tuple[EnsembleRetriever, List]:
        """검색기를 로드하거나 새로 구축합니다."""
        db_dir = str(settings.CHROMA_DB_DIR)
        store_path = str(settings.DOCSTORE_PATH)

        if not force_rebuild and os.path.exists(db_dir) and os.path.exists(store_path):
            return self._load_existing_retriever(db_dir, store_path, k)
        
        return self._build_new_retriever(pdf_paths, db_dir, store_path, k)

    def _load_existing_retriever(self, db_dir: str, store_path: str, k: int) -> Tuple[EnsembleRetriever, List]:
        """저장된 하이브리드 인덱스(VectorDB, Docstore)를 불러옵니다."""
        print("⚡ 하드디스크에 저장된 DB와 문서를 불러옵니다!")
        vectorstore = Chroma(persist_directory=db_dir, embedding_function=self.embedding_model)
        
        with open(store_path, "rb") as f:
            store, all_documents = pickle.load(f)

        parent_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore, docstore=store, 
            child_splitter=self.child_splitter, search_kwargs={"k": k}
        )
        
        bm25_retriever = BM25Retriever.from_documents(all_documents)
        bm25_retriever.k = k

        hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, parent_retriever], weights=[0.5, 0.5])
        return hybrid_retriever, all_documents

    def _build_new_retriever(self, pdf_paths: List[str], db_dir: str, store_path: str, k: int) -> Tuple[EnsembleRetriever, List]:
        """원본 PDF로부터 텍스트를 추출해 부분 단위로 나누고 하이브리드 검색기를 구축합니다."""
        print(f"⏳ {len(pdf_paths)}개의 문서를 기반으로 하이브리드 DB를 구축합니다...")
        all_documents = []
        for path in pdf_paths:
            loader = PyMuPDFLoader(path)
            docs = loader.load()
            file_name = os.path.basename(path)
            for doc in docs:
                page_num = doc.metadata.get("page", 0) + 1
                doc.page_content = f"[문서: {file_name}, {page_num}페이지]\n" + doc.page_content
            all_documents.extend(docs)

        vectorstore = Chroma(persist_directory=db_dir, embedding_function=self.embedding_model)
        store = InMemoryStore()
        parent_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore, docstore=store,
            child_splitter=self.child_splitter, search_kwargs={"k": k}
        )

        # Batch Insertion
        batch_size = 100
        for i in range(0, len(all_documents), batch_size):
            parent_retriever.add_documents(all_documents[i : i + batch_size])
            print(f"   🔄 진행 중... ({min(i + batch_size, len(all_documents))}/{len(all_documents)} 완료)")

        with open(store_path, "wb") as f:
            pickle.dump((store, all_documents), f)

        bm25_retriever = BM25Retriever.from_documents(all_documents)
        bm25_retriever.k = k
        
        hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, parent_retriever], weights=[0.5, 0.5])
        return hybrid_retriever, all_documents
