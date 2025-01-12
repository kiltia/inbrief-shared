from datetime import timedelta
from typing import ClassVar, Tuple
from uuid import UUID

from pydantic import BaseModel


class Entity(BaseModel):
    pass


class Request(Entity):
    chat_id: int
    request_id: UUID
    request_type: str
    status: str
    config_id: int
    time_passed: timedelta

    _table_name: ClassVar[str] = "request"


class StorySource(Entity):
    story_id: UUID
    source_id: int
    channel_id: int

    _table_name: ClassVar[str] = "story_source"


class StorySources(Entity):
    story_id: UUID
    request_id: UUID
    text: str
    date: str
    reference: str
    embeddings: str

    _table_name: ClassVar[str] = "story_sources"


class RequestSource(Entity):
    request_id: UUID
    source_id: UUID

    _table_name: ClassVar[str] = "request_source"


class Story(Entity):
    story_id: UUID
    request_id: UUID
    category_id: UUID

    _table_name: ClassVar[str] = "story"
    _pk: ClassVar[str] = "story_id"


class Preset(Entity):
    preset_id: UUID
    preset_name: str
    chat_folder_link: str
    editor_prompt: str
    date_created: str
    inactive: bool = False

    _table_name: ClassVar[str] = "preset"
    _pk: ClassVar[str] = "preset_id"


class User(Entity):
    chat_id: int

    _table_name: ClassVar[str] = "users"
    _pk: ClassVar[str] = "chat_id"


class UserPreset(Entity):
    chat_id: int
    preset_id: UUID

    _table_name: ClassVar[str] = "user_preset"


class UserPresets(Entity):
    preset_id: UUID
    chat_folder_link: str
    editor_prompt: str
    preset_name: str
    inactive: bool

    _table_name: ClassVar[str] = "user_presets"


class Config(Entity):
    config_id: int
    embedding_source: str
    linking_method: str
    categorize_method: str | None = None
    summary_method: str
    editor_model: str
    inactive: bool

    _table_name: ClassVar[str] = "config"
    _pk: ClassVar[str] = "config_id"


class Summary(Entity):
    summary_id: UUID
    chat_id: int
    story_id: UUID
    summary: str
    title: str
    density: str
    config_id: int
    feedback: str | None = None
    date_created: str

    _table_name: ClassVar[str] = "summary"
    _pk: ClassVar[Tuple[str, str]] = ("summary_id", "density")


class Callback(Entity):
    callback_id: UUID
    callback_data: str

    _table_name: ClassVar[str] = "callback"
    _pk: ClassVar[str] = "callback_id"
