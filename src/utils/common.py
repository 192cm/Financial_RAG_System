def clean_llm_response(response) -> str:
    """LLM(Gemini 등)이 멀티모달 처리 시 리스트 형태로 반환하는 응답에서 텍스트만 깔끔하게 추출합니다.

    Args:
        response: llm.invoke().content 등에서 반환된 결과 (str 또는 List)
        
    Returns:
        str: 최종 텍스트 답변
    """
    if isinstance(response, list):
        # [{ 'type': 'text', 'text': '...' }] 형태 대응
        return response[0].get("text", "") if isinstance(response[0], dict) else str(response[0])
    
    # 이미 문자열인 경우 그대로 반환
    return response
