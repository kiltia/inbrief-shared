import logging
from datetime import datetime
from typing import List
from uuid import UUID, uuid4

from asgi_correlation_id import CorrelationIdMiddleware
from databases import Database
from fastapi import FastAPI
from matcher import Matcher
from ranking import Ranker, init_scorers

from shared.db import PgRepository, create_db_string
from shared.entities import Source, Story, StoryPost
from shared.logger import configure_logging
from shared.models import LinkingRequest
from shared.resources import SharedResources
from shared.routes import LinkerRoutes
from shared.utils import DB_DATE_FORMAT, SHARED_CONFIG_PATH

app = FastAPI()
app.add_middleware(CorrelationIdMiddleware, validator=None)


logger = logging.getLogger("app")


class Context:
    def __init__(self):
        self.shared_resources = SharedResources(
            f"{SHARED_CONFIG_PATH}/settings.json"
        )
        self._pg = Database(create_db_string(self.shared_resources.pg_creds))
        self.story_repository = PgRepository(self._pg, Story)
        self.story_post_repository = PgRepository(self._pg, StoryPost)
        self.ranker = None

    async def init_db(self):
        await self._pg.connect()

    async def dispose_db(self):
        await self._pg.disconnect()


ctx = Context()


@app.post(LinkerRoutes.GET_STORIES)
async def get_stories(request: LinkingRequest):
    matcher = Matcher(
        request.entities,
        request.embedding_source,
        request.scorer,
        request.metric,
    )
    stories_nums = matcher.get_stories(
        method_name=request.method, **request.config
    )

    uuids = [
        uuid4() for _ in range(len(stories_nums) + len(stories_nums[-1]) - 1)
    ]

    stories_uuids = [Story(story_id=i) for i in uuids]
    await ctx.story_repository.add(stories_uuids)

    entities = []
    stories: List[tuple[UUID, List[Source]]] = []
    uuid_num = 0
    for i in range(len(stories_nums[:-1])):
        stories.append((uuids[uuid_num], []))
        for j in range(len(stories_nums[i])):
            source = request.entities[stories_nums[i][j]]
            entity = StoryPost(
                story_id=uuids[uuid_num],
                source_id=source.source_id,
                channel_id=source.channel_id,
            )
            entities.append(entity)
            stories[i][1].append(source)

        uuid_num += 1
    # NOTE(sokunkov): We need to finally decide what we want to do
    # with the noisy cluster
    for i in range(len(stories_nums[-1])):
        stories.append((uuids[uuid_num], []))
        source = request.entities[stories_nums[-1][i]]
        entity = StoryPost(
            story_id=uuids[uuid_num],
            source_id=source.source_id,
            channel_id=request.entities[stories_nums[-1][i]].channel_id,
        )
        entities.append(entity)
        stories[-1][1].append(source)
        uuid_num += 1

    await ctx.story_post_repository.add(entities)

    entities = stories_nums[:-1]
    entities.extend(stories_nums[-1])

    weights = ctx.shared_resources.config.ranking.weights

    stories = list(
        map(
            lambda t: (
                t[0],
                sorted(
                    t[1],
                    key=lambda x: datetime.strptime(x.date, DB_DATE_FORMAT),
                ),
            ),
            stories,
        )
    )

    stories = ctx.ranker.get_sorted(
        stories, request.required_scorers, weights=weights
    )

    story_ids = list(map(lambda t: t[0], stories))
    return story_ids


@app.get("/")
def hello_world():
    return "Linker API is running!"


@app.on_event("shutdown")
async def dispose():
    await ctx.dispose_db()


@app.on_event("startup")
async def main() -> None:
    configure_logging()
    ctx.ranker = Ranker(init_scorers())

    await ctx.init_db()
