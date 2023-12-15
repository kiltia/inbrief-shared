import logging
from typing import List

import openai
from asgi_correlation_id import CorrelationIdMiddleware
from databases import Database
from embedders import init_embedders
from fastapi import FastAPI, Response, status
from telethon import TelegramClient
from utils import SESSION_PATH

from config import Credentials
from scraper import parse_channels, retrieve_channels
from shared.db import PgRepository, create_db_string
from shared.entities import Folder, Source
from shared.logger import configure_logging
from shared.models import ParseRequest, SyncRequest
from shared.resources import SharedResources
from shared.routes import ScraperRoutes
from shared.utils import SHARED_CONFIG_PATH

logger = logging.getLogger("app")

app = FastAPI()
app.add_middleware(CorrelationIdMiddleware, validator=None)


class Context:
    def __init__(self):
        self.creds = Credentials()
        self.client = TelegramClient(
            f"{SESSION_PATH}/{self.creds.session}",
            self.creds.api_id,
            self.creds.api_hash,
        )
        self.shared_settings = SharedResources(
            f"{SHARED_CONFIG_PATH}/settings.json"
        )
        self.pg = Database(create_db_string(self.shared_settings.pg_creds))
        self.folder_repository = PgRepository(self.pg, Folder)
        self.source_repository = PgRepository(self.pg, Source)

    async def init_db(self):
        await self.pg.connect()

    async def dispose_db(self):
        await self.pg.disconnect()


ctx = Context()


@app.post(ScraperRoutes.PARSE)
async def parse(request: ParseRequest, response: Response) -> List[Source]:
    logger.info("Started serving scrapping request")
    entities = await parse_channels(ctx, **request.model_dump())
    # TODO(nrydanov): Need to add caching there in case all posts for required
    # time period are already stored in database (#137)
    if entities:
        await ctx.source_repository.add(entities, ignore_conflict=True)
        logger.debug("Data was saved to database successfully")
    else:
        response.status_code = status.HTTP_204_NO_CONTENT
    return entities


@app.post(ScraperRoutes.SYNC)
async def sync(request: SyncRequest):
    logger.debug("Started serving fetch request")
    response = await retrieve_channels(ctx, request.chat_folder_link)
    return response


@app.on_event("startup")
async def main() -> None:
    configure_logging()
    logger.info("Started loading embedders")
    init_embedders(ctx.shared_settings.components.embedders)
    openai.api_key = ctx.shared_settings.openai_api_key
    await ctx.init_db()
    await ctx.client.start()


@app.on_event("shutdown")
async def disconnect() -> None:
    await ctx.client.disconnect()
    await ctx.dispose_db()
