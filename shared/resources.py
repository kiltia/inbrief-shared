# NOTE(nrydanov): Probably, this file doesn't have proper name for its content

from typing import Dict, List

from pydantic import BaseModel
from pydantic_settings import BaseSettings

from shared.models import JSONSettings


class DatabaseConfig(BaseModel):
    driver: str
    url: str
    port: int
    db_name: str


class ComponentSettings(BaseModel):
    embedders: List[str]
    classifier: str | None
    summarizing: List[str]


class Ranking(BaseModel):
    weights: Dict[str, float]


class Configuration(BaseModel):
    category_async_pool_size: int
    summarize_async_pool_size: int
    short_ops_timeout: int
    long_ops_timeout: int
    ranking: Ranking


class SharedResources(JSONSettings):
    components: ComponentSettings
    pg_creds: DatabaseConfig
    config: Configuration

    def __init__(self, path: str):
        super().__init__(path)


class EnvironmentSettings(BaseSettings):
    pass
