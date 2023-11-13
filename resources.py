# NOTE(nrydanov): Probably, this file doesn't have proper name for its content

from typing import List

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


class SharedResources(JSONSettings):
    components: ComponentSettings
    pg_creds: DatabaseCredentials
    openai_api_key: str
