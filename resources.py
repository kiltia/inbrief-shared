# NOTE(nrydanov): Probably, this file doesn't have proper name for its content

from typing import Dict, List

from pydantic import BaseModel

from shared.models import JSONSettings


class DatabaseCredentials(BaseModel):
    driver: str
    username: str
    password: str
    url: str
    port: int
    db_name: str


class ComponentSettings(BaseModel):
    embedders: List[str]
    summarizing: List[str]


class Ranking(BaseModel):
    weights: Dict[str, float]


class Configuration(BaseModel):
    summarize_thread_pool: int
    short_ops_timeout: int
    long_ops_timeout: int
    ranking: Ranking


class SharedResources(JSONSettings):
    components: ComponentSettings
    pg_creds: DatabaseCredentials
    config: Configuration

    def __init__(self, path: str):
        super().__init__(path)
