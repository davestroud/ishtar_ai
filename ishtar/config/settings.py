from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    env: str = Field(default="dev", validation_alias="ENV")
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")

    vllm_base_url: str | None = Field(default=None, validation_alias="VLLM_BASE_URL")
    tgi_base_url: str | None = Field(default=None, validation_alias="TGI_BASE_URL")

    embeddings_provider: str = Field(default="openai", validation_alias="EMBEDDINGS_PROVIDER")
    vector_backend: str = Field(default="faiss", validation_alias="VECTOR_BACKEND")
    index_name: str = Field(default="ishtar-articles", validation_alias="INDEX_NAME")

    max_context_tokens: int = Field(default=4000, validation_alias="MAX_CONTEXT_TOKENS")
    rerank_top_k: int = Field(default=20, validation_alias="RERANK_TOP_K")
    retrieve_k: int = Field(default=12, validation_alias="RETRIEVE_K")

    class Config:
        env_file = ".env"

settings = Settings()
