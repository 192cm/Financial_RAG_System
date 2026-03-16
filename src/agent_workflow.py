import os
import re

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.vision_utils import get_page_text_and_image


def run_corrective_agent(
    query, pdf_path, start_page_num, all_documents, max_expansions=3
):
    """
    [범용 금융 에이전트]
    특정 기업이나 항목에 종속되지 않고, 질문(Query)의 맥락을 스스로 분석하여
    기업명 검증, 페이지 확장, 정답 도출을 수행합니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)

    current_pages = [start_page_num, start_page_num + 1, start_page_num + 2]
    pdf_filename = os.path.basename(pdf_path)

    # 발행 연도 추론 (일반화)
    year_match = re.search(r"\d{4}", pdf_filename)
    report_year = year_match.group() if year_match else "미상"

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

        # 🎯 [완전 일반화된 범용 지능형 프롬프트]
        prompt = f"""당신은 방대한 기업 공시 자료에서 사용자가 원하는 정보를 정확히 찾아내는 전문 데이터 분석가입니다.

[분석 환경 정보]
- 현재 검토 중인 파일명: {pdf_filename}
- 사용자 질문: {query}
- 텍스트 컨텍스트: {text_context}

[행동 지침 - 0순위 무조건 준수]
1. 문서 검증 (기업명 일치):
   - 사용자 [질문]의 타겟 기업과 현재 [파일명/본문]의 기업이 다르면, 즉시 "WRONG_DOCUMENT" 라고만 출력하세요.

2. 질문 의도 및 최적 출처 판단 (Generalization):
   - 질문이 요구하는 정보의 성격을 스스로 파악하고 가장 공식적인 출처인지 판단하세요.
   
★ 2-1. 엄격한 조건 일치 (Strict Constraint):
   - 사용자 [질문]에서 **특정 표 이름(예: '연결 현금흐름표', '재무상태표')이나 특정 섹션을 명확히 지목했다면**, 현재 분석 중인 표가 반드시 그 표와 일치해야 합니다.
   - 만약 표 이름이 다르다면(예: 질문은 현금흐름표인데 현재 표는 배당지표인 경우), **설령 그 안에 정답(당기순이익 등)이 완벽하게 적혀 있더라도 절대 채택하지 마세요.**
   - 조건이 어긋나면 즉시 "NOT_FOUND_IN_THIS_CANDIDATE"를 출력하세요.

3. 무관한 페이지 패스:
   - 현재 페이지에 질문과 관련된 직접적이고 명확한 정답이 없다면 "NOT_FOUND_IN_THIS_CANDIDATE"를 출력하세요. 억지로 유추하지 마세요.

4. 윈도우 확장 (Sliding Window):
   - 정답이 포함된 표(Table)나 문단이 현재 페이지 하단에서 끊겨 다음 페이지로 이어진다고 판단되면 "NEXT_PAGE_NEEDED"를 출력하세요.

5. 최종 정답 도출 (출처 명시):
   - 정답을 확실히 찾았다면, 반드시 [문서명, 페이지 번호, 해당 섹션/표 이름]을 명시하여 답변을 작성하세요.
"""

        content_list = [{"type": "text", "text": prompt}]
        for b64 in collected_images:
            content_list.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )

        print(f"🤔 Gemini가 질문의 의도를 분석하며 검증 중...")
        response = llm.invoke([HumanMessage(content=content_list)])

        # 응답 텍스트 추출
        if isinstance(response.content, list):
            answer_text = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in response.content
            ).strip()
        else:
            answer_text = str(response.content).strip()

        # 🛑 범용화된 에이전트 의사결정 로직
        if "WRONG_DOCUMENT" in answer_text:
            print(f"❌ 결과: 기업명 불일치 (Wrong Document - {pdf_filename})")
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

        # 위 부정 키워드 3개가 없다면 정답으로 간주
        print("✅ 결과: 정답 발견!")
        return answer_text

    return "NOT_FOUND_IN_THIS_CANDIDATE"
