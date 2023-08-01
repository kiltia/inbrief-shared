from pydantic_settings import BaseSettings


class ElasticSettings(BaseSettings):
    host: str
    password: str
    ca_certs: str
    index_name: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
