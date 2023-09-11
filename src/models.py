from typing import List

from pydantic import BaseModel

from shared.utils import DEFAULT_END_DATE


class ParserRequest(BaseModel):
    channels: List[str]
    required_embedders: List[str] | None = None
    offset_date: str | None = None
    end_date: str = DEFAULT_END_DATE
    social: bool = False
    markup: bool = False
    classify_model: str = "gpt-3.5-turbo"
    max_retries: int = 1
