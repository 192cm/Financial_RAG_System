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

    def run(self, query: str, pdf_path: str, start_page_num: int, all_documents: List, max_expansions: int = 3) -> str:
        current_pages = [start_page_num, start_page_num + 1, start_page_num + 2]
        pdf_filename = os.path.basename(pdf_path)

        for step in range(max_expansions + 1):
            print(f"🔄 [Agent Loop {step+1}] 분석 중인 페이지: {[p+1 for p in current_pages]}")
            
            text_context, images = self._collect_data(pdf_path, current_pages, all_documents)
            
            prompt = self._get_prompt(pdf_filename, query, text_context)
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
            elif status == "SUCCESS":
                return answer
            elif status == "NOT_FOUND_IN_THIS_CANDIDATE":
                return "NOT_FOUND_IN_THIS_CANDIDATE"
        
        return "NOT_FOUND_IN_THIS_CANDIDATE"

    def _collect_data(self, pdf_path: str, pages: List[int], all_docs: List):
        texts, images = [], []
        for p in pages:
            t, img = get_page_text_and_image(pdf_path, p, all_docs)
            if t: texts.append(f"[페이지 {p+1} 텍스트]\n{t}")
            if img: images.append(img)
        return "\n\n".join(texts), images

    def _get_prompt(self, filename: str, query: str, context: str) -> str:
        return f"""전문 데이터 분석가로서 아래 정보를 바탕으로 질문에 답하세요.
[분석 파일]: {filename}
[사용자 질문]: {query}
[텍스트 컨텍스트]: {context}

[행동 지침]
1. 기업명이 다르면 즉시 "WRONG_DOCUMENT" 출력.
2. 질문에서 지목한 특정 표가 아니면 "NOT_FOUND_IN_THIS_CANDIDATE" 출력.
3. 정보가 다음 장에 이어질 것 같으면 "NEXT_PAGE_NEEDED" 출력.
4. 정답을 찾으면 상세 수치와 출처를 포함해 "SUCCESS"와 함께 답변 출력.

반드시 아래 JSON 형식으로만 응답하세요:
{{
    "status": "(WRONG_DOCUMENT, NEXT_PAGE_NEEDED, NOT_FOUND_IN_THIS_CANDIDATE, SUCCESS 중 택 1)",
    "answer": "정답 내용 또는 빈 문자열"
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
