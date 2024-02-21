import logging
import os
from contextlib import asynccontextmanager

from asgi_correlation_id import CorrelationIdMiddleware
from classifiers import get_classifier
from databases import Database
from embedders import init_embedders
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from telethon import TelegramClient
from utils import SESSION_PATH

from config import Credentials
from openai_api import get_async_client
from scraper import parse_channels, retrieve_channels
from shared.db import PgRepository, create_db_string
from shared.entities import Channel, Folder, Source
from shared.logger import configure_logging
from shared.models import ParseRequest, ParseResponse
from shared.resources import SharedResources
from shared.routes import ScraperRoutes
from shared.utils import SHARED_CONFIG_PATH


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    logger.info("Started loading embedders")
    init_embedders(ctx.shared_settings.components.embedders)
    await ctx.init_db()
    await ctx.client.start()
    yield
    await ctx.client.disconnect()
    await ctx.dispose_db()


logger = logging.getLogger("scraper")

app = FastAPI(lifespan=lifespan)
app.add_middleware(CorrelationIdMiddleware, validator=None)


class Context:
    def __init__(self):
        self.creds = Credentials()
        self.client = TelegramClient(
            f"{SESSION_PATH}/{self.creds.session}",
            self.creds.api_id,
            self.creds.api_hash,
            system_version="4.16.30-vxCUSTOM",
        )
        self.openai_client = get_async_client(os.getenv("OPENAI_API_KEY"))
        self.shared_settings = SharedResources(
            f"{SHARED_CONFIG_PATH}/settings.json"
        )
        self.classifier = get_classifier(
            self.shared_settings.components.classifier
        )
        self.pg = Database(create_db_string(self.shared_settings.pg_creds))
        self.folder_repository = PgRepository(self.pg, Folder)
        self.source_repository = PgRepository(self.pg, Source)
        self.channel_repository = PgRepository(self.pg, Channel)

    async def init_db(self):
        await self.pg.connect()

    async def dispose_db(self):
        await self.pg.disconnect()


ctx = Context()


@app.post(ScraperRoutes.PARSE)
async def parse(request: ParseRequest) -> ParseResponse:
    logger.info("Started serving scrapping request")
    entities, skipped_channel_ids = await parse_channels(
        ctx, **request.model_dump()
    )
    # TODO(nrydanov): Need to add caching there in case all posts for required
    # time period are already stored in database (#137)
    if entities:
        await ctx.source_repository.add(entities, ignore_conflict=True)
        logger.debug("Data was saved to database successfully")
        return ParseResponse(
            sources=entities,
            skipped_channel_ids=skipped_channel_ids,
        )
    else:
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT)


@app.get(ScraperRoutes.SYNC)
async def sync(link: str):
    logger.debug("Started serving sync request")
    response = await retrieve_channels(ctx, link)
    return response
