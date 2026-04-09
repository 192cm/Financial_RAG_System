import re
from src.models import get_gemini_model

def llm_as_a_judge(query: str, pred: str, gt_text: str) -> float:
    """LLM 심판을 통해 정답의 정확성과 무결성을 1.0~5.0 사이로 평가합니다."""
    if not pred or "실패" in pred: return 1.0
    
    llm = get_gemini_model(temperature=0)
    prompt = f"""당신은 금융 RAG 시스템의 결과를 평가하는 공정한 심판입니다.
[질문]: {query}
[정답]: {gt_text}
[답변]: {pred}

위 내용을 비교하여 점수를 매겨주세요.
1. 먼저 답변이 정답의 핵심 내용을 포함하고 있는지, 수치적 오류는 없는지, 출처가 명확한지 분석하세요.
2. 분석을 바탕으로 1.0~5.0점 사이의 점수를 부여하세요.

출력 형식:
Score: [점수]
Reason: [이유]

평가 기준:
- [5.0] 완벽한 수치(단위 포함), 정확한 연도와 출처
- [3.0-4.0] 전반적으로 맞으나 출처가 누락되었거나 문장이 불완전한 경우
- [2.0 이하] 수치가 틀렸거나, 단위 환산 오류, 혹은 반올림으로 인한 부정확함 (금융 데이터에서 수치 오류는 치명적 오답으로 간주)
- [1.0] 환각(Hallucination) 또는 질문과 무관한 답변
"""
    import time
    max_judge_attempts = 3
    for attempt in range(max_judge_attempts):
        try:
            res = llm.invoke(prompt).content
            if isinstance(res, list):
                res = res[0].get("text", "") if isinstance(res[0], dict) else str(res[0])
            
            # Score: [점수] 패턴에서 숫자 추출
            match = re.search(r'Score:\s*([0-9.]+)', res, re.IGNORECASE)
            if match:
                score_str = match.group(1)
            else:
                score_str = re.sub(r'[^0-9.]', '', res.split('\n')[0]) # 실패 시 첫 줄에서 숫자만 추출 시도
                
            return max(1.0, min(5.0, float(score_str)))
        except Exception as e:
            if attempt < max_judge_attempts - 1:
                print(f"   ⚖️ 심판 오류: {e}. {10*(attempt+1)}초 후 재시도...")
                time.sleep(10 * (attempt + 1))
            else:
                print(f"   ⚖️ 심판 최종 실패: {e}")
                return 1.0
