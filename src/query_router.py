import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_company_list_from_directory(data_dir="../data/raw"):
    """
    data/raw 폴더의 PDF 파일명에서 기업명 목록을 동적으로 자동 추출합니다.
    """
    company_list = []

    # 폴더가 존재하는지 확인
    if not os.path.exists(data_dir):
        print(f"⚠️ 경고: '{data_dir}' 경로를 찾을 수 없습니다.")
        return company_list

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".pdf"):
            # 1. 대괄호 형태 추출 (예: [삼성SDI]사업보고서... -> 삼성SDI)
            match = re.search(r"\[(.*?)\]", filename)
            if match:
                company_name = match.group(1)
            else:
                # 2. 언더바 형태 추출 (예: 삼성전자_사업보고서... -> 삼성전자)
                company_name = filename.split("_")[0]

            # 중복 방지 (이미 리스트에 있으면 넣지 않음)
            if company_name not in company_list:
                company_list.append(company_name)

    return company_list


def extract_target_company(query, company_list=None, data_dir="../data/raw"):
    """
    사용자의 질문(Query)에서 타겟 기업명을 추출하는 지능형 자동 라우터.
    """
    # 파라미터로 기업 리스트를 안 넘기면 폴더에서 실시간으로 읽어옴!
    if company_list is None:
        company_list = get_company_list_from_directory(data_dir)

    # 만약 폴더가 비어있거나 에러가 났을 때를 대비한 최소한의 방어 로직
    if not company_list:
        company_list = ["삼성전자"]
        print("⚠️ 기업 목록을 불러오지 못해 기본값으로 세팅합니다.")

    companies_str = ", ".join(company_list)

    llm_router = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)

    router_prompt = ChatPromptTemplate.from_template(
        "당신은 금융 검색어 분석기입니다. 사용자의 [질문]에서 타겟으로 하는 '기업명'을 추출하세요.\n"
        f"우리 DB에 있는 기업: [{companies_str}]\n"
        "위 기업 중 하나가 포함되어 있다면 오직 그 '기업명'만 출력하고, 특정 기업을 지칭하지 않았다면 'ALL'이라고 출력하세요.\n\n"
        "[질문]: {query}"
    )

    company_extractor = router_prompt | llm_router | StrOutputParser()

    print(
        f"🤔 [Self-Query Router] 폴더에서 자동 인식한 {len(company_list)}개 기업 중 필터 조건을 탐색합니다..."
    )
    extracted_company = company_extractor.invoke({"query": query}).strip()

    return extracted_company
