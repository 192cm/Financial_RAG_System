from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from src.vision_utils import get_page_text_and_image


def run_corrective_agent(
    query, pdf_path, start_page_num, all_documents, max_expansions=10
):
    """
    N, N+1, N+2 페이지를 기본으로 주입하고, 필요시 N+3, N+4로 동적 확장하는 에이전트 루프
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # 기본 슬라이딩 윈도우 세팅 (N, N+1, N+2)
    current_pages = [start_page_num, start_page_num + 1, start_page_num + 2]
    final_answer = ""

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

        # 교정적(Corrective) 프롬프트: 텍스트와 이미지를 교차 검증하도록 지시
        if step < max_expansions:
            prompt = f"""당신은 날카로운 금융 데이터 분석가입니다.
[전체 텍스트 컨텍스트]를 참고하여, 첨부된 {len(current_pages)}장의 [표 이미지들]을 교차 검증하세요.

[전체 텍스트 컨텍스트 (참고용)]
{text_context}

[질문]: {query}

[행동 지침]
1. 텍스트 컨텍스트에서 질문에 대한 힌트가 있는지 확인하세요.
2. 텍스트에 힌트가 있다면, 첨부된 이미지 표 안에 해당 수치가 정확히 있는지 확인하세요.
3. 텍스트엔 내용이 이어지는데, 마지막 이미지의 표 밑바닥이 잘려 최종 정답을 도출할 수 없다면, 억지로 유추하지 말고 오직 "NEXT_PAGE_NEEDED" 라고만 출력하세요.
4. 이미지 안에서 완벽한 정답을 확인했다면 명확한 문장으로 답변하세요.
"""
        else:
            prompt = f"""제공된 텍스트와 이미지들을 모두 교차 검증하여, 현재까지 파악한 정보만으로 최대한 정확히 답변하세요.\n[질문]: {query}"""

        content_list = [{"type": "text", "text": prompt}]
        for b64 in collected_images:
            content_list.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )

        print("🤔 Gemini 2.5 Flash가 다중 모달 데이터를 교차 검증 중입니다...")
        response = llm.invoke([HumanMessage(content=content_list)])
        answer_text = response.content.strip()

        if "NEXT_PAGE_NEEDED" in answer_text:
            print(
                "🚨 AI 평가: '표가 더 이어집니다.' -> 윈도우를 다음 페이지로 동적 확장합니다."
            )
            current_pages.append(current_pages[-1] + 1)
        else:
            print(
                "✅ AI 평가: '현재 문맥 내에 완벽한 정답이 존재합니다.' -> 탐색 완료."
            )
            final_answer = answer_text
            break

    return final_answer
