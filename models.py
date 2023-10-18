from enum import Enum
from typing import List

from pydantic import BaseModel

from shared.utils import DEFAULT_END_DATE


class PresetData(BaseModel):
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
    # NOTE(nrydanov): Isn't expected to give end-user possibility to choose
    # some of those parameters, but it's required for now
    embedding_source: EmbeddingSource
    linking_method: LinkingMethod
    summary_method: SummaryMethod


class Payload(BaseModel):
    # TODO(nrydanov): Change str for dates to datetime.time validation (#78)
    end_date: str
    offset_date: str | None = None
    preset_data: PresetData


class Request(BaseModel):
    config: Config
    payload: Payload
    required_density: List[Density]


class ParserRequest(BaseModel):
    chat_folder_link: str
    required_embedders: List[str] | None = None
    offset_date: str | None = None
    end_date: str = DEFAULT_END_DATE
    social: bool = False
    markup: bool = False
    classify_model: str = "gpt-3.5-turbo"
    max_retries: int = 1


class LinkingRequest(BaseModel):
    texts: List[str]
    embeddings: List[List[float]]
    dates: List[str]
    method: LinkingMethod
    config: dict


class SummaryRequest(BaseModel):
    story: List[str]
    method: SummaryMethod = SummaryMethod.BART
    density: Density
