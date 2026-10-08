from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field

class LLMSettings(BaseModel):
    provider: str = Field(default="google")
    generator_model: str = Field(default="gemini-1.5-flash")
    judge_model: str = Field(default="gemini-1.5-pro")
    embedding_model: str = Field(default="models/embedding-001")
    openai_api_key: Optional[str] = Field(default=None)
    google_api_key: Optional[str] = Field(default=None)

class VectorStoreSettings(BaseModel):
    path: str = Field(default="./data/vector_store")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)

class JudgeSettings(BaseModel):
    pass_threshold: float = Field(default=8.0)
    max_regeneration_attempts: int = Field(default=2)
    weight_faithfulness: float = Field(default=0.30)
    weight_correctness: float = Field(default=0.25)
    weight_relevance: float = Field(default=0.20)
    weight_completeness: float = Field(default=0.15)
    weight_clarity: float = Field(default=0.10)

class APISettings(BaseModel):
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=True)

class DatabaseSettings(BaseModel):
    url: str = Field(default="sqlite:///./data/chatbot.db")

class LogSettings(BaseModel):
    level: str = Field(default="INFO")
    file: str = Field(default="./logs/app.log")

class Settings(BaseSettings):
    llm_provider: str = Field(default="google", alias="LLM_PROVIDER")
    generator_model: str = Field(default="gemini-1.5-flash", alias="GENERATOR_MODEL")
    judge_model: str = Field(default="gemini-1.5-pro", alias="JUDGE_MODEL")
    embedding_model: str = Field(default="models/embedding-001", alias="EMBEDDING_MODEL")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    
    vector_store_path: str = Field(default="./data/vector_store", alias="VECTOR_STORE_PATH")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    
    judge_pass_threshold: float = Field(default=8.0, alias="JUDGE_PASS_THRESHOLD")
    max_regeneration_attempts: int = Field(default=2, alias="MAX_REGENERATION_ATTEMPTS")
    
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    debug: bool = Field(default=True, alias="DEBUG")
    
    database_url: str = Field(default="sqlite:///./data/chatbot.db", alias="DATABASE_URL")
    
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="./logs/app.log", alias="LOG_FILE")

    @property
    def llm(self) -> LLMSettings:
        return LLMSettings(
            provider=self.llm_provider,
            generator_model=self.generator_model,
            judge_model=self.judge_model,
            embedding_model=self.embedding_model,
            openai_api_key=self.openai_api_key,
            google_api_key=self.google_api_key
        )

    @property
    def vector_store(self) -> VectorStoreSettings:
        return VectorStoreSettings(
            path=self.vector_store_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    @property
    def judge(self) -> JudgeSettings:
        return JudgeSettings(
            pass_threshold=self.judge_pass_threshold,
            max_regeneration_attempts=self.max_regeneration_attempts
        )
        
    @property
    def api(self) -> APISettings:
        return APISettings(
            host=self.api_host,
            port=self.api_port,
            debug=self.debug
        )
        
    @property
    def db(self) -> DatabaseSettings:
        return DatabaseSettings(
            url=self.database_url
        )
        
    @property
    def log(self) -> LogSettings:
        return LogSettings(
            level=self.log_level,
            file=self.log_file
        )

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

@lru_cache()
def get_settings() -> Settings:
    return Settings()
