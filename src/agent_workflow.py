import os
import re

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.vision_utils import get_page_text_and_image


def run_corrective_agent(
    query, pdf_path, start_page_num, all_documents, max_expansions=3
):
    """
    [The Final Financial Agent]
    기업명 검증 -> 섹션 판단 -> 동의어 탐색 -> 산술 연산 금지 로직이 통합된 범용 에이전트.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)

    current_pages = [start_page_num, start_page_num + 1, start_page_num + 2]
    pdf_filename = os.path.basename(pdf_path)

    year_match = re.search(r"\d{4}", pdf_filename)
    report_year = year_match.group() if year_match else "미상"
    target_year = str(int(report_year) - 1) if report_year != "미상" else "당기"

    for step in range(max_expansions + 1):
        display_pages = [p + 1 for p in current_pages]
        print(f"\n🔄 [Agent Loop {step+1}] 분석 중인 페이지: {display_pages}")

        collected_texts = []
        collected_images = []

        for p in current_pages:
            t, img = get_page_text_and_image(pdf_path, p, all_documents)
            if t:
                collected_texts.append(f"[페이지 {p+1} 텍스트]\n{t}")
            if img:
                collected_images.append(img)

        text_context = "\n\n".join(collected_texts)

        # 🎯 [통합 지능형 프롬프트]
        prompt = f"""당신은 무결성을 최우선으로 하는 금융 데이터 감사관입니다.

[분석 환경 정보]
- 파일명: {pdf_filename}
- 보고서 발행 연도: {report_year}년 (이 보고서의 '당기'는 보통 {target_year}년 실적을 의미합니다.)
- 사용자 질문: {query}
- 이미지 및 텍스트 문맥: {text_context}


[행동 지침]
1. 현재 조사 중인 문서({pdf_filename})가 질문에서 요구한 '대상 기업'의 보고서인지 최우선으로 확인하세요.
- 파일명은 영어로 작성되어 있으니 대상 기업을 영어로 번역한 후 확실하게 확인하세요.
- 만약 질문은 '삼성SDI'인데 파일명이 'samsung_electronics'이거나, 이미지 내용이 다른 기업 것이라면 아무것도 묻지도 따지지도 말고 "WRONG_DOCUMENT"라고만 답하세요.
- 주의: 주석 표에 나열된 '계열사(종속기업)' 이름에 속지 마세요. 보고서의 본체가 질문의 기업과 다르다면(예: 삼성전자를 묻는데 삼성SDI 보고서인 경우) 즉시 "WRONG_DOCUMENT"라고 답변하세요.
2. 텍스트 컨텍스트에서 질문에 대한 힌트가 있는지 확인하세요.
- 주의: 몇 년도에 관한 질문인지 확실하게 인지하세요.
- 주의: 질문과 관련된 단어와 동의어를 잘 파악하세요. (예: 당기순이익, 당기손이익은 동의어)
3. 텍스트에 힌트가 있다면, 첨부된 이미지 표 안에 해당 수치가 정확히 있는지 확인하세요.
4. 텍스트엔 내용이 이어지는데, 마지막 이미지의 표 밑바닥이 잘려 최종 정답을 도출할 수 없다면, 억지로 유추하지 말고 오직 "NEXT_PAGE_NEEDED" 라고만 출력하세요.
5. 이미지 안에서 완벽한 정답을 확인했다면 명확한 문장으로 답변하세요.


"""

        content_list = [{"type": "text", "text": prompt}]
        for b64 in collected_images:
            content_list.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )

        print(f"🤔 Gemini 3.1 Flash Lite가 검증 중...")
        response = llm.invoke([HumanMessage(content=content_list)])

        # 안전한 응답 추출
        if isinstance(response.content, list):
            answer_text = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in response.content
            ).strip()
        else:
            answer_text = str(response.content).strip()

        # 에이전트 의사결정
        if "WRONG_DOCUMENT" in answer_text:
            print("❌ 결과: 기업명 불일치 (Wrong Document)")
            return "WRONG_DOCUMENT"

        if "NEXT_PAGE_NEEDED" in answer_text:
            if step >= max_expansions:
                print("⚠️ 확장 한도 초과")
                return "NOT_FOUND_IN_THIS_CANDIDATE"
            print(f"🚨 결과: 표 이어짐 -> {current_pages[-1] + 2}p로 확장")
            current_pages.append(current_pages[-1] + 1)
            continue

        if "NOT_FOUND_IN_THIS_CANDIDATE" in answer_text:
            print("⏭️ 결과: 해당 위치 정답 없음 (Skip)")
            return "NOT_FOUND_IN_THIS_CANDIDATE"

        print("✅ 결과: 정답 발견!")
        return answer_text

    return "NOT_FOUND_IN_THIS_CANDIDATE"
