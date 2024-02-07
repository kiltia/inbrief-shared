import logging
from contextlib import asynccontextmanager

from asgi_correlation_id import CorrelationIdMiddleware
from databases import Database
from fastapi import FastAPI
from matcher import Matcher

from shared.db import PgRepository, create_db_string
from shared.entities import Story, StorySource
from shared.logger import configure_logging
from shared.models import LinkingRequest
from shared.resources import SharedResources
from shared.routes import LinkerRoutes
from shared.utils import SHARED_CONFIG_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    await ctx.init_db()
    yield
    await ctx.dispose_db()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CorrelationIdMiddleware, validator=None)


logger = logging.getLogger("app")


class Context:
    def __init__(self):
        self.shared_resources = SharedResources(
            f"{SHARED_CONFIG_PATH}/settings.json"
        )
        self._pg = Database(create_db_string(self.shared_resources.pg_creds))
        self.story_repository = PgRepository(self._pg, Story)
        self.story_post_repository = PgRepository(self._pg, StorySource)
        self.ranker = None

    async def init_db(self):
        await self._pg.connect()

    async def dispose_db(self):
        await self._pg.disconnect()


ctx = Context()


@app.post(LinkerRoutes.GET_STORIES)
async def get_stories(request: LinkingRequest):
    matcher = Matcher(
        request.entries,
        request.config.embedding_source,
        request.config.scorer,
        request.config.metric,
    )

    results, embeddings = matcher.get_stories(
        method_name=request.config.method,
        **request.settings,
        return_plot_data=request.return_plot_data,
    )

    if request.return_plot_data:
        return {"results": results, "embeddings": embeddings.tolist()}
    else:
        return {"results": results}


@app.get("/")
def hello_world():
    return "Linker API is running!"
