from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import settings

def get_gemini_model(model_name: str = "gemini-3.1-flash-lite-preview", temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    """언어 모델(Gemini) 인스턴스를 중앙에서 생성합니다.
    503(Service Unavailable) 및 Rate Limit 대응을 위해 기본 재시도 횟수를 늘렸습니다."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=settings.get_api_key(),
        max_retries=10,  
        timeout=120      # 응답 지연 시 타임아웃 120초 설정
    )
