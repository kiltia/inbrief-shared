import logging

import openai
import pandas as pd
from config import Credentials
from embedders import init_embedders
from fastapi import FastAPI
from telethon import TelegramClient
from utils import SESSION_PATH

from scraper import parse_channels_by_links
from shared.models import ParserRequest
from shared.utils import LOGGING_FORMAT

logger = logging.getLogger(__name__)

creds = Credentials()
app = FastAPI()

session_path = f"{SESSION_PATH}/{creds.session}"
client = TelegramClient(session_path, creds.api_id, creds.api_hash)


@app.post("/scraper/")
async def parse(request: ParserRequest):
    logger.info("Started serving scrapping request")
    response = await parse_channels_by_links(client, **request.model_dump())
    logger.info("Saving data to dictionary")
    df = pd.DataFrame.from_dict(response)
    df.dropna(inplace=True)
    df.to_json("output.json")
    return df.to_dict("list")


@app.on_event("startup")
async def main() -> None:
    logging.basicConfig(
        format=LOGGING_FORMAT,
        datefmt="%m-%d %H:%M:%S",
        level=logging.INFO,
        force=True,
    )
    logger.info("Started loading embedders")
    init_embedders()
    openai.api_key = creds.openai_api_key
    await client.start()


@app.on_event("shutdown")
async def disconnect() -> None:
    await client.disconnect()
