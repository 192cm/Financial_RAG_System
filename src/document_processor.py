import os
import pickle
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore


def build_parent_retriever(pdf_paths, chunk_size=400, overlap=40):
    """
    다중 PDF 문서를 로드하여 Parent Document Retriever를 구축합니다.
    (대용량 처리를 위한 Batch 인서트 및 캐싱 기능 탑재)
    """
    db_dir = "../data/chroma_db"
    store_path = "../data/docstore.pkl"

    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )

    # 1. 캐싱된 DB가 있다면 1초 만에 로드!
    if os.path.exists(db_dir) and os.path.exists(store_path):
        print("⚡ 하드디스크에 저장된 DB와 문서를 1초 만에 불러옵니다!")
        vectorstore = Chroma(
            persist_directory=db_dir, embedding_function=embedding_model
        )

        with open(store_path, "rb") as f:
            store, all_documents = pickle.load(f)

        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            search_kwargs={"k": 2},
        )
        return retriever, all_documents

    # 2. 캐싱된 DB가 없다면 최초 구축 (배치 처리 적용!)
    print(f"⏳ {len(pdf_paths)}개의 문서를 로드하고 Vector DB를 최초 구축합니다...")

    all_documents = []
    for path in pdf_paths:
        loader = PyMuPDFLoader(path)
        all_documents.extend(loader.load())

    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embedding_model)
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        search_kwargs={"k": 2},
    )

    # ==========================================
    # 🚀 [핵심 수정] 대용량 데이터를 위한 Batch Insert 로직
    # ==========================================
    batch_size = 100  # 한 번에 100페이지(Parent)씩 쪼개서 DB에 넣습니다.
    total_pages = len(all_documents)

    print(
        f"📦 총 {total_pages}페이지의 문서를 {batch_size}개씩 묶어서 안전하게 DB에 삽입합니다..."
    )

    for i in range(0, total_pages, batch_size):
        batch_docs = all_documents[i : i + batch_size]
        retriever.add_documents(batch_docs)
        # 진행률 출력
        current_done = min(i + batch_size, total_pages)
        print(f"   🔄 진행 중... ({current_done}/{total_pages} 페이지 완료)")

    # 3. 완성된 DB와 문서를 하드디스크에 저장(얼리기)
    with open(store_path, "wb") as f:
        pickle.dump((store, all_documents), f)

    print("✅ 대용량 다중 문서 기반 DB 구축 및 하드디스크 저장 완료!")

    return retriever, all_documents
