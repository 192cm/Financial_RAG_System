import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

class Settings:
    """금융 RAG 시스템 통합 설정 관리 클래스"""
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    CONFIG_DIR = BASE_DIR / "config"
    RAW_DATA_DIR = DATA_DIR / "raw"
    CHROMA_DB_DIR = DATA_DIR / "chroma_db"
    DOCSTORE_PATH = DATA_DIR / "docstore.pkl"
    VISION_INDEX_NAME = "multi_doc_vision_index"
    
    def __init__(self, env_path: str = None):
        load_dotenv(dotenv_path=env_path or self.BASE_DIR / ".env")
        
        with open(self.CONFIG_DIR / "config.yaml", "r", encoding="utf-8") as f:
            self._yaml_config = yaml.safe_load(f)
            
        self.GOOGLE_API_KEY = os.getenv(self._yaml_config['api_keys']['google_gemini_env_name'])
        if self.GOOGLE_API_KEY:
            os.environ["GOOGLE_API_KEY"] = self.GOOGLE_API_KEY

    def get_api_key(self) -> str:
        """설정된 Google Gemini API Key를 반환합니다."""
        return self.GOOGLE_API_KEY

settings = Settings()
