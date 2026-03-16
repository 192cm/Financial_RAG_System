import os
import pickle
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


def build_parent_retriever(
    pdf_paths, chunk_size=400, overlap=40, force_rebuild=False, k=10
):
    """
    다중 PDF 문서를 로드하여 하이브리드 검색기(Vector + BM25)를 구축합니다.
    (메타데이터 주입, 대용량 Batch 인서트 및 캐싱 기능 탑재)
    """
    db_dir = "../data/chroma_db"
    store_path = "../data/docstore.pkl"

    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )

    # ==========================================
    # 1. 캐싱된 DB 로드 (force_rebuild가 False일 때)
    # ==========================================
    if not force_rebuild and os.path.exists(db_dir) and os.path.exists(store_path):
        print("⚡ 하드디스크에 저장된 DB와 문서를 불러옵니다!")
        vectorstore = Chroma(
            persist_directory=db_dir, embedding_function=embedding_model
        )

        with open(store_path, "rb") as f:
            store, all_documents = pickle.load(f)

        # Vector 리트리버 복원
        parent_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            search_kwargs={"k": k},  # 알맹이에 k값 지정
        )

        # BM25 리트리버 복원 (저장된 all_documents 활용)
        print("🔍 BM25 키워드 검색기 복원 중...")
        bm25_retriever = BM25Retriever.from_documents(all_documents)
        bm25_retriever.k = k  # 알맹이에 k값 지정

        # 앙상블 리트리버 결합
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, parent_retriever], weights=[0.5, 0.5]
        )

        print("✅ 하이브리드 검색기(캐시) 로드 완료!")
        return hybrid_retriever, all_documents

    # ==========================================
    # 2. DB 최초 구축 (또는 force_rebuild=True 일 때)
    # ==========================================
    print(f"⏳ {len(pdf_paths)}개의 문서를 로드하고 하이브리드 DB를 새로 구축합니다...")

    all_documents = []
    for path in pdf_paths:
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        file_name = os.path.basename(path)

        # 🚀 [핵심 로직] 메타데이터 강제 주입 (Metadata Injection)
        # 청크가 쪼개져도 어느 파일의 몇 페이지인지 모델이 알 수 있도록 텍스트 맨 앞에 태깅
        for doc in docs:
            page_num = doc.metadata.get("page", 0) + 1
            doc.page_content = (
                f"[문서: {file_name}, {page_num}페이지]\n" + doc.page_content
            )

        all_documents.extend(docs)

    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embedding_model)
    store = InMemoryStore()

    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        search_kwargs={"k": k},  # 알맹이에 k값 지정
    )

    # 🚀 [대용량 처리] Batch Insert 로직
    batch_size = 100
    total_pages = len(all_documents)
    print(
        f"📦 총 {total_pages}페이지의 문서를 {batch_size}개씩 묶어서 DB에 삽입합니다..."
    )

    for i in range(0, total_pages, batch_size):
        batch_docs = all_documents[i : i + batch_size]
        parent_retriever.add_documents(batch_docs)
        current_done = min(i + batch_size, total_pages)
        print(f"   🔄 진행 중... ({current_done}/{total_pages} 페이지 완료)")

    # 완성된 Store와 문서를 하드디스크에 저장
    with open(store_path, "wb") as f:
        pickle.dump((store, all_documents), f)

    # 🚀 [하이브리드 로직] BM25 구축 및 결합
    print("🔍 BM25 키워드 검색기를 구축합니다...")
    bm25_retriever = BM25Retriever.from_documents(all_documents)
    bm25_retriever.k = k

    print("🤝 Vector와 BM25를 결합한 하이브리드 검색기 완성!")
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, parent_retriever], weights=[0.5, 0.5]
    )

    print("✅ 대용량 다중 문서 기반 하이브리드 DB 구축 및 저장 완료!")
    return hybrid_retriever, all_documents
