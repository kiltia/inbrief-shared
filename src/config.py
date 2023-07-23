from pydantic_settings import BaseSettings


class Credentials(BaseSettings):
    session: str
    api_hash: str
    api_id: str
    openai_api_key: str

    class Config:
        env_file = "./config/.env"
        env_file_encoding = "utf-8"


class Settings(BaseSettings):
    classify_model: str = "gpt-3.5-turbo"
    max_retries: int = 1

    class Config:
        env_file = "config/parser.cfg"
        env_file_encoding = "utf-8"
