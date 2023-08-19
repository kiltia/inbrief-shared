from enum import Enum
from typing import List

from pydantic import BaseModel


class LinkingMethod(Enum):
    DBSCAN = "dbscan"
    BM25 = "bm25"


class LinkingRequest(BaseModel):
    texts: List[str]
    embeddings: List[List[float]]
    dates: List[str]
    method: LinkingMethod
    config: dict
