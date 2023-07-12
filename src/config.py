from pydantic_settings import BaseSettings
from utils import DEFAULT_END_DATE


class Credentials(BaseSettings):
    phone: str
    api_hash: str
    api_id: str
    openai_api_key: str

    class Config:
        env_file = "config/.env"
        env_file_encoding = "utf-8"


class ParserSettings(BaseSettings):
    num_workers: int
    offset_date: str
    channels: list
    categories: list
    end_date: str = DEFAULT_END_DATE
    social: bool = False
    markup: bool = False
    model: str = "gpt-3.5-turbo"
    max_retries: int = 1

    class Config:
        env_file = "config/parser.cfg"
        env_file_encoding = "utf-8"
