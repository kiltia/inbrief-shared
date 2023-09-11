from typing import List

from pydantic import BaseModel

from shared.models import LinkingMethod


class LinkingRequest(BaseModel):
    texts: List[str]
    embeddings: List[List[float]]
    dates: List[str]
    method: LinkingMethod
    config: dict
