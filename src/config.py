from pydantic_settings import BaseSettings


class Credentials(BaseSettings):
    session: str = ""
    api_hash: str
    api_id: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
