from typing import ClassVar, List
from uuid import UUID

from pydantic import BaseModel


class Entity(BaseModel):
    pass


class Channel(BaseModel):
    channel_id: int
    title: str
    about: str
    subscribers: int

    _table_name: ClassVar[str] = "channel"
    _pk: ClassVar[str] = "channel_id"


class Folder(Entity):
    chat_folder_link: str
    channels: List[int]

    _table_name: ClassVar[str] = "folder"
    _pk: ClassVar[str] = "chat_folder_link"


class Source(Entity):
    source_id: int
    text: str
    date: str
    channel_id: int
    reference: str
    embeddings: str
    label: str | None = None
    comments: list | None = None
    reactions: str | None = None
    views: int

    _table_name: ClassVar[str] = "source"
    _pk: ClassVar[str] = "source_id"


class StoryPost(Entity):
    story_id: UUID
    source_id: int
    channel_id: int

    _table_name: ClassVar[str] = "story_post"


class StoryPosts(Entity):
    story_id: UUID
    text: str
    date: str
    reference: str

    _table_name: ClassVar[str] = "story_posts"


class Story(Entity):
    story_id: UUID

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
    cur_preset: UUID | None = None

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
    feedback: int | None = None
    date_created: str

    _table_name: ClassVar[str] = "summary"


class Callback(Entity):
    callback_id: UUID
    callback_data: str

    _table_name: ClassVar[str] = "callback"
    _pk: ClassVar[str] = "callback_id"
