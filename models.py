import json
from enum import Enum
from typing import List
from uuid import UUID

from pydantic import BaseModel


class JSONSettings(BaseModel):
    def __init__(self, path: str):
        with open(path, "r") as f:
            config_data = json.load(f)
            return super().__init__(**config_data)

    class Config:
        populate_by_name = True


class PresetData(BaseModel):
    preset_name: str
    chat_folder_link: str
    editor_prompt: str

    @classmethod
    def get_fields(cls):
        return list(cls.__fields__.keys())


class SummaryType(str, Enum):
    STORYLINES = "storylines"
    SINGLE_NEWS = "single_news"


class EmbeddingSource(str, Enum):
    FTMLM = "ft+mlm"
    OPENAI = "openai"
    MLM = "mlm"


class LinkingMethod(str, Enum):
    DBSCAN = "dbscan"
    BM25 = "bm25"
    NO_LINKER = "no_linker"


class SummaryMethod(str, Enum):
    OPENAI = "openai"
    BART = "bart"


class Density(str, Enum):
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


class LinkingRequest(BaseRequest):
    text: List[str]
    embeddings: List[List[float]]
    date: List[str]
    source_id: List[int]
    channel_id: List[int]
    method: LinkingMethod
    config: dict


# Summarizer API


class SummaryRequest(BaseRequest):
    summary_id: UUID
    story_id: UUID
    config_id: int
    chat_id: int
    density: Density


# Editor


class EditorRequest(BaseRequest):
    summary_id: UUID
    model: str = "gpt-4-1106-preview"
    style: str = "влиятельный политик"
    density: Density


# Supervisor API


class SummarizeRequest(ExternalRequest):
    config_id: int
    story_id: str
    required_density: List[Density]


class FetchRequest(ExternalRequest):
    end_date: str
    offset_date: str | None = None


class CallbackPostRequest(BaseRequest):
    callback_data: dict


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
