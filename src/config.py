from pydantic_settings import BaseSettings


class LinkerAPISettings(BaseSettings):
    get_stories: str
    default_method: str
    available_methods: list

    class Config:
        env_file = "config/app_conf.cfg"
        env_file_encoding = "utf-8"


class DefaultClusteringSettings(BaseSettings):
    eps: float
    min_samples: int
    metric: str

    class Config:
        env_file = "config/clustering_conf.cfg"
        env_file_encoding = "utf-8"


class DefaultSearcherSettings(BaseSettings):
    depth: int
    min_samples: int
    threshold: float
    semantic_threshold: list

    class Config:
        env_file = "config/searcher_conf.cfg"
        env_file_encoding = "utf-8"
