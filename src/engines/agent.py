import json
import os
import re
from typing import List, Optional, Any
from langchain_core.messages import HumanMessage
from src.models import get_gemini_model
from src.utils.vision import get_page_text_and_image

class CorrectiveAgent:
    """금융 문서 검증 및 정답 도출을 수행하는 에이전트 클래스"""
    
    def __init__(self, model_name: str = "gemini-3.1-flash-lite-preview"):
        self.llm = get_gemini_model(model_name=model_name)

    def run(self, query: str, pdf_path: str, initial_pages: List[int], all_documents: List, max_expansions: int = 3, table_name: str = "NONE") -> str:
        # 초기 페이지 그룹으로 시작 → NEXT_PAGE_NEEDED 시 마지막 페이지 기준으로 누적 확장
        current_pages = list(initial_pages)
        pdf_filename = os.path.basename(pdf_path)

        for step in range(max_expansions + 1):
            print(f"🔄 [Agent Loop {step+1}] 분석 중인 페이지: {[p+1 for p in current_pages]}")
            
            text_context, images = self._collect_data(pdf_path, current_pages, all_documents)
            
            prompt = self._get_prompt(pdf_filename, query, text_context, table_name)
            content_list = [{"type": "text", "text": prompt}]
            for b64 in images:
                content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            
            response = self.llm.invoke([HumanMessage(content=content_list)])
            status, answer = self._parse_response(response.content)

            if status == "WRONG_DOCUMENT":
                return "WRONG_DOCUMENT"
            elif status == "NEXT_PAGE_NEEDED":
                if step < max_expansions:
                    current_pages.append(current_pages[-1] + 1)
                    continue
                else:
                    # 최대 확장 횟수 초과 → 이 후보에서는 찾을 수 없음
                    return "NOT_FOUND_IN_THIS_CANDIDATE"
            elif status == "SUCCESS":
                return answer
            elif status == "NOT_FOUND_IN_THIS_CANDIDATE":
                return "NOT_FOUND_IN_THIS_CANDIDATE"
            else:
                # UNKNOWN(파싱 실패) → 이 후보 포기
                return "NOT_FOUND_IN_THIS_CANDIDATE"

        return "NOT_FOUND_IN_THIS_CANDIDATE"

    def _collect_data(self, pdf_path: str, pages: List[int], all_docs: List):
        texts, images = [], []
        for p in pages:
            t, img = get_page_text_and_image(pdf_path, p, all_docs)
            if t: texts.append(f"[페이지 {p+1} 텍스트]\n{t}")
            if img: images.append(img)
        return "\n\n".join(texts), images

    def _get_prompt(self, filename: str, query: str, context: str, table_name: str = "NONE") -> str:
        # 표 이름이 명시된 경우 프롬프트에 강제 검증 규칙 삽입
        table_rule = (
            f'\n0. [표 검증 - 최우선] 현재 페이지에 "{table_name}" 표(또는 섹션)가 명확히 존재하지 않으면 '
            f'→ "NOT_FOUND_IN_THIS_CANDIDATE" (다른 표의 동일 항목값 사용 절대 금지)'
            if table_name and table_name != "NONE" else ""
        )
        return f"""당신은 금융 문서 전문 분석가입니다. 아래 규칙을 반드시 순서대로 따르세요.

[분석 파일]: {filename}
[사용자 질문]: {query}
[텍스트 컨텍스트]:
{context}

[판단 규칙 - 번호 순서대로 우선 적용]
{table_rule}
1. [기업명 불일치] 컨텍스트의 기업명이 질문의 기업명과 다르면 → "WRONG_DOCUMENT"

2. [표 단절 감지] 아래 중 하나라도 해당하면 → "NEXT_PAGE_NEEDED"
   - 표의 행이 "(다음 쪽에 계속)", "계속" 등으로 끝남
   - 표의 마지막 행에 합계(total)나 소계 행이 없이 중간에 끊김
   - 질문에서 요구하는 특정 항목이 표 헤더에는 있으나 값 행이 페이지 내에 없음
   - 텍스트가 문장 중간에서 잘림

3. [표 불일치] 질문이 지목한 특정 표가 아닌 다른 표(요약표, 별도재무제표 등)라면 → "NOT_FOUND_IN_THIS_CANDIDATE"

4. [정답 확정] 질문이 요구하는 수치를 원문에서 직접 찾은 경우 → "SUCCESS"
   - 반드시 원문에 명시된 숫자만 사용 (계산·추론 금지)
   - 연도(예: 2024년)와 기수(예: 제56기)를 원문 그대로 표기
   - 단위(백만원, 천원, 원 등)를 원문 그대로 표기
   - 답변은 질문에 대한 완결된 문장으로 작성하고, 반드시 출처(파일명, 페이지 번호)를 명시하세요. (예: "삼성전자의 2024년 당기순이익은 [분석 파일]의 [페이지 n]에 기재된 바와 같이 34,451,351백만원입니다.")

반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
{{
    "status": "WRONG_DOCUMENT | NEXT_PAGE_NEEDED | NOT_FOUND_IN_THIS_CANDIDATE | SUCCESS",
    "answer": "SUCCESS일 때만 '파일명과 페이지가 포함된 완결된 문장'으로 답변 작성, 나머지는 빈 문자열"
}}
"""

    def _parse_response(self, response_text: Any):
        try:
            res = response_text
            if isinstance(res, list):
                res = res[0].get("text", "") if isinstance(res[0], dict) else str(res[0])
            
            clean_json = str(res).strip("` \n")
            if clean_json.startswith("json\n"): clean_json = clean_json[5:]
            parsed = json.loads(clean_json)
            return parsed.get("status", "UNKNOWN"), parsed.get("answer", "")
        except:
            return "UNKNOWN", ""
