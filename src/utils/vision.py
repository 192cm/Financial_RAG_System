import fitz
import base64


def get_page_text_and_image(pdf_path, p_num, all_documents):
    """
    특정 소스(PDF)와 물리적 페이지 번호(p_num)에 해당하는 전체 텍스트와 고화질 이미지를 반환합니다.
    """
    # 1. 원본 텍스트 추출 (all_documents 활용)
    text = ""
    for doc in all_documents:
        if doc.metadata.get("source") == pdf_path and doc.metadata.get("page") == p_num:
            text = doc.page_content
            break

    # 2. 고화질 이미지 렌더링 및 Base64 인코딩
    b64_img = None
    try:
        pdf_doc = fitz.open(pdf_path)
        if p_num < len(pdf_doc):
            page = pdf_doc.load_page(p_num)
            # 2.0 배율로 선명하게 캡처 (표 무손실 보존)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            b64_img = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    except Exception as e:
        print(f"이미지 변환 오류: {e}")

    return text, b64_img
