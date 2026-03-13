import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore


def build_parent_retriever(pdf_paths, chunk_size=400, overlap=40):
    """
    다중 PDF 문서를 로드하여 Parent Document Retriever를 구축합니다.
    """
    print(f"⏳ {len(pdf_paths)}개의 문서를 로드하고 Vector DB를 구축합니다...")

    all_documents = []
    for path in pdf_paths:
        loader = PyMuPDFLoader(path)
        all_documents.extend(loader.load())  # 1페이지 = 1 Parent Document

    # 바늘(키워드)을 찾기 위한 정밀한 자식 분할기
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )

    # 한국어 특화 임베딩 모델
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

    # Vector DB와 메모리 창고 세팅
    vectorstore = Chroma(
        collection_name="parent_store", embedding_function=embedding_model
    )
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        search_kwargs={"k": 2},
    )

    retriever.add_documents(all_documents)
    print("✅ 다중 문서 기반 하이브리드 DB 구축 완료!")

    return retriever, all_documents
