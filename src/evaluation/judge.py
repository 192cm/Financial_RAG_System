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

위 내용을 비교하여 1.0~5.0점 사이의 점수만 출력하세요 (예: 4.5).
- 완벽한 정답과 출처: 5.0
- 부분적 틀림/출처 없음: 2.0-4.0
- 환각/오답: 1.0
"""
    import time
    max_judge_attempts = 3
    for attempt in range(max_judge_attempts):
        try:
            res = llm.invoke(prompt).content
            if isinstance(res, list):
                res = res[0].get("text", "") if isinstance(res[0], dict) else str(res[0])
            score_str = re.sub(r'[^0-9.]', '', res)
            return max(1.0, min(5.0, float(score_str)))
        except Exception as e:
            if attempt < max_judge_attempts - 1:
                print(f"   ⚖️ 심판 오류: {e}. {10*(attempt+1)}초 후 재시도...")
                time.sleep(10 * (attempt + 1))
            else:
                print(f"   ⚖️ 심판 최종 실패: {e}")
                return 1.0
