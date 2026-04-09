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

    def run(self, query: str, pdf_path: str, initial_pages: List[int], all_documents: List, max_expansions: int = 3, table_name: str = "NONE", return_metadata: bool = False) -> Any:
        """초기 검색된 문서 페이지들을 바탕으로 정답을 찾고 필요 시 주변 페이지로 검색 범위를 확장합니다."""
        # 초기 페이지 그룹으로 시작 → NEXT/PREV 시 해당 방향으로 누적 확장
        current_pages = list(initial_pages)
        pdf_filename = os.path.basename(pdf_path)
        
        # Context Carry-over (루프 중간에 발견된 중요 메타데이터 기억)
        metadata_context = ""
        
        # 토큰 사용량 통계 초기화
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for step in range(max_expansions + 1):
            current_pages = sorted(list(set(current_pages))) # 중복 제거 및 정렬
            print(f"[Agent Loop {step+1}] 분석 중인 페이지: {[p+1 for p in current_pages]}")
            
            text_context, images = self._collect_data(pdf_path, current_pages, all_documents)
            
            prompt = self._get_prompt(pdf_filename, query, text_context, table_name, metadata_context)
            content_list = [{"type": "text", "text": prompt}]
            for b64 in images:
                content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            
            response = self.llm.invoke([HumanMessage(content=content_list)])
            
            # 토큰 사용량 누적
            usage = getattr(response, "usage_metadata", {})
            for k in total_usage:
                if k in usage:
                    total_usage[k] += usage[k]

            status, answer, meta = self._parse_response(response.content)
            
            # 메타데이터 업데이트 (단위 정보 등 보존)
            if meta:
                metadata_context = meta

            if status == "WRONG_DOCUMENT":
                final_status = "WRONG_DOCUMENT"
                break
            elif status == "NEXT_PAGE_NEEDED":
                if step < max_expansions:
                    current_pages.append(current_pages[-1] + 1)
                    continue
                else:
                    final_status = "NOT_FOUND_IN_THIS_CANDIDATE"
                    break
            elif status == "PREV_PAGE_NEEDED":
                if step < max_expansions:
                    if current_pages[0] > 0:
                        current_pages.insert(0, current_pages[0] - 1)
                        continue
                    else:
                        # 더 이상 이전 페이지가 없음 → 현재 컨텍스트에서 최선 다하기
                        pass
                else:
                    final_status = "NOT_FOUND_IN_THIS_CANDIDATE"
                    break
            elif status == "SUCCESS":
                final_status = "SUCCESS"
                break
            else:
                final_status = "NOT_FOUND_IN_THIS_CANDIDATE"
                break
        else:
            final_status = "NOT_FOUND_IN_THIS_CANDIDATE"

        # 결과 조립
        res_answer = answer if final_status == "SUCCESS" else final_status
        
        metadata = {"usage": total_usage}
        return {"answer": res_answer, "metadata": metadata} if return_metadata else res_answer

    def _collect_data(self, pdf_path: str, pages: List[int], all_docs: List):
        """에이전트가 분석할 대상 페이지들의 텍스트와 이미지 데이터를 수집합니다."""
        texts, images = [], []
        for p in pages:
            t, img = get_page_text_and_image(pdf_path, p, all_docs)
            if t: texts.append(f"[페이지 {p+1} 텍스트]\n{t}")
            if img: images.append(img)
        return "\n\n".join(texts), images

    def _get_prompt(self, filename: str, query: str, context: str, table_name: str = "NONE", metadata_context: str = "") -> str:
        """에이전트 판단을 위해 기업명, 표 제약조건, 정합성을 검증하는 프롬프트를 생성합니다."""
        table_rule = (
            f'\n0. [표 검증 - 최우선] 현재 페이지에 "{table_name}" 표(또는 섹션)가 명확히 존재하지 않으면 '
            f'→ "NOT_FOUND_IN_THIS_CANDIDATE" (다른 표의 동일 항목값 사용 절대 금지)'
            if table_name and table_name != "NONE" else ""
        )
        
        meta_info = f"\n[이전 루프에서 파악된 정보]: {metadata_context}" if metadata_context else ""

        return f"""당신은 금융 문서 전문 분석가입니다. 아래 규칙을 반드시 순서대로 따르세요.

[분석 파일]: {filename}
[사용자 질문]: {query}{meta_info}
[텍스트 컨텍스트]:
{context}

[판단 규칙 - 번호 순서대로 우선 적용]
{table_rule}
1. [기업명 불일치] 컨텍스트의 기업명이 질문의 기업명과 다르면 → "WRONG_DOCUMENT"

2. [표/컨텍스트 단절 감지] 아래 중 하나라도 해당하면 확장을 요청하세요:
   - "NEXT_PAGE_NEEDED": 표의 행이 아래로 이어지거나, 합계/소계 없이 중간에 끊김, 텍스트가 문장 중간에서 잘림
   - "PREV_PAGE_NEEDED": 현재 정답 수치는 찾았으나 '단위(예: 백만원)' 정보가 상단에 없고, 표의 헤더가 시작되지 않은 채 중간부터 나타남

3. [표 불일치] 질문이 지목한 특정 표가 아닌 다른 표라면 → "NOT_FOUND_IN_THIS_CANDIDATE"

4. [정답 확정] 질문이 요구하는 '정확한 수치'를 원문에서 직접 찾고 '단위'까지 확신할 수 있는 경우에만 → "SUCCESS"
   - [주의] 관련 항목(예: 연구개발비, 매출액)은 있으나 질문이 요구하는 '비율'이나 '계산된 값'이 원문에 명시되어 있지 않은 경우 → 절대로 SUCCESS를 사용하지 말고 "NOT_FOUND_IN_THIS_CANDIDATE"를 반환하세요.
   - [연도-기수 매칭 가이드] 
     * 질문의 연도와 표 헤더의 연도가 정확히 일치하는지 확인하십시오.
     * 주의: 2025년 보고서(제 57기)에서 2024년 값을 찾을 경우, '당기'가 아닌 '전기' 헤더(제 56기) 아래의 값을 추출해야 합니다. 
     * 단순히 '현재 기수'의 값을 2024년 값으로 답변하는 실수를 절대 하지 마십시오.
   - 반드시 원문에 명시된 숫자만 사용 (계산·추론 금지)
   - 반드시 단위(백만원, 천원 등)를 원문 그대로 표기 (이전 루프 정보 참고 가능)
   - 답변 끝에 반드시 "[최종 정답]: 수치+단위" 형식으로 정답만 별도로 표기하세요.
     * 예: "2025년 당기순이익은 1,234백만원입니다. [최종 정답]: 1,234백만원"
   - 답변은 질문에 대한 완결된 문장으로 작성하고 출처를 명시하세요.

반드시 아래 JSON 형식으로만 응답하세요:
{{
    "status": "WRONG_DOCUMENT | NEXT_PAGE_NEEDED | PREV_PAGE_NEEDED | NOT_FOUND_IN_THIS_CANDIDATE | SUCCESS",
    "answer": "SUCCESS일 때만 작성, 나머지는 빈 문자열",
    "found_metadata": "현재까지 파악된 단위(Unit) 정보 등 (다음 루프를 위해 기록)"
}}
"""

    def _parse_response(self, response_text: Any):
        """에이전트의 LLM 응답값에서 검증결과(status), 답변(answer), 메타데이터를 파싱합니다."""
        try:
            res = response_text
            if isinstance(res, list):
                res = res[0].get("text", "") if isinstance(res[0], dict) else str(res[0])
            
            clean_json = str(res).strip("` \n")
            if clean_json.startswith("json\n"): clean_json = clean_json[5:]
            
            # JSON만 추출하기 위해 re 사용 (간혹 LLM이 JSON 앞뒤에 텍스트를 붙이는 경우 대비)
            json_match = re.search(r'\{.*\}', clean_json, re.DOTALL)
            if json_match:
                clean_json = json_match.group()

            parsed = json.loads(clean_json)
            if not isinstance(parsed, dict):
                return "UNKNOWN", "", ""

            return (
                parsed.get("status", "UNKNOWN"), 
                parsed.get("answer", ""), 
                parsed.get("found_metadata", "")
            )
        except:
            return "UNKNOWN", "", ""
