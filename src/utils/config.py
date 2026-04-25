import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3-8b-8192")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "customer_support_kb")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))
    TOP_K: int = int(os.getenv("TOP_K", 4))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.60))
    ESCALATION_KEYWORDS: list = [
        "legal", "lawsuit", "sue", "fraud", "hacked", "data breach", "refund abuse"
    ]
 