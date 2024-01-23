import json
from enum import Enum
from typing import List
from uuid import UUID

from pydantic import BaseModel

from shared.entities import Source


class JSONSettings(BaseModel):
    def __init__(self, path: str):
        with open(path, "r") as f:
            config_data = json.load(f)
            return super().__init__(**config_data)

    class Config:
        populate_by_name = True


def extend_enum(inherited_enum):
    def wrapper(added_enum):
        joined = {}
        for item in inherited_enum:
            joined[item.name] = item.value
        for item in added_enum:
            joined[item.name] = item.value
        return Enum(added_enum.__name__, joined)

    return wrapper


class PresetData(BaseModel):
    preset_name: str
    chat_folder_link: str
    editor_prompt: str

    @classmethod
    def get_fields(cls):
        return list(cls.model_fields.keys())


class EmbeddingSource(str, Enum):
    FTMLM = "ft+mlm"
    OPENAI = "openai"
    MLM = "mlm"


class ClusteringMethod(str, Enum):
    HDBSCAN = "hdbscan"
    DBSCAN = "dbscan"
    KMeans = "kmeans"
    AffinityPropagation = "affinity_propagation"
    OPTICS = "optics"
    SpectralClustering = "spectral_clustering"
    AgglomerativeClustering = "agglomerative_clustering"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


@extend_enum(ClusteringMethod)
class LinkingMethod(str, Enum):
    NO_LINKER = "no_linker"


class OpenAIModels(str, Enum):
    GPT3_TURBO = "gpt-3.5-turbo-1106"
    GPT4 = "gpt-4"
    GPT4_32k = "gpt-4-32k"
    GPT4_TURBO = "gpt-4-1106-preview"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class SummaryMethod(str, Enum):
    OPENAI = "openai"
    BART = "bart"


class Density(str, Enum):
    TITLE = "title"
    SMALL = "small"
    LARGE = "large"

    def next(self):
        cls = self.__class__
        members = list(cls)
        index = members.index(self) + 1
        if index >= len(members):
            raise StopIteration("End of enumeration reached")
        return members[index]

    def prev(self):
        cls = self.__class__
        members = list(cls)
        index = members.index(self) - 1
        if index < 0:
            raise StopIteration("Beginning of enumeration reached")
        return members[index]


class Config(BaseModel):
    embedding_source: EmbeddingSource
    linking_method: LinkingMethod
    summary_method: SummaryMethod


class BaseRequest(BaseModel):
    pass


class ExternalRequest(BaseRequest):
    chat_id: int


# Scraper API


class ParseRequest(BaseRequest):
    channels: List[int]
    required_embedders: List[str] | None = None
    offset_date: str | None = None
    end_date: str
    social: bool = False


class SyncRequest(BaseRequest):
    chat_folder_link: str


# Linker API


class LinkingScorers(str, Enum):
    SILHOUETTE = "silhouette"


class DistancesMetrics(str, Enum):
    CITYBLOCK = "cityblock"
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    HAVERSINE = "haversine"


class LinkingRequest(BaseRequest):
    entities: List[Source]
    method: LinkingMethod
    embedding_source: EmbeddingSource
    scorer: LinkingScorers
    metric: DistancesMetrics
    config: dict
    required_scorers: List[str] | None = None


# Summarizer API


class EditorConfig(BaseModel):
    style: str
    model: OpenAIModels


class SummaryConfig(BaseModel):
    summary_model: OpenAIModels
    editor_config: EditorConfig | None = None


class SummaryRequest(BaseRequest):
    story: list[str]
    summary_method: SummaryMethod
    config: SummaryConfig
    density: Density


# Supervisor API


class SummarizeRequest(ExternalRequest):
    config_id: int
    story_id: str
    required_density: List[Density]


class FetchRequest(ExternalRequest):
    end_date: str
    offset_date: str | None = None
    social: bool = False
    config_id: int | None = None


class CallbackPostRequest(BaseRequest):
    callback_data: dict


class ConfigPostRequest(BaseRequest):
    config_id: int
    embedding_source: str
    linking_method: str
    summary_method: str
    editor_model: str


class CallbackPatchRequest(BaseRequest):
    callback_id: UUID
    callback_data: dict


class UserRequest(ExternalRequest):
    chat_id: int


class ChangePresetRequest(BaseRequest):
    cur_preset: UUID


class PartialPresetUpdate(ExternalRequest):
    preset_id: UUID
    preset_name: str | None = None
    chat_folder_link: str | None = None
    editor_prompt: str | None = None
    inactive: bool | None = None
