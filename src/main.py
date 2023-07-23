import logging

import openai
import pandas as pd
from config import Credentials
from embedders import init_embedders
from fastapi import FastAPI
from models import ParserRequest
from telethon import TelegramClient
from utils import LOGGING_FORMAT, SESSION_PATH

from scraper import parse_channels_by_links

logger = logging.getLogger(__name__)

creds = Credentials()
app = FastAPI()


@app.post("/scraper/")
async def parse(payload: ParserRequest):
    session_path = f"{SESSION_PATH}/{creds.session}"
    logger.info("Started serving scrapping request")
    async with TelegramClient(session_path, creds.api_id, creds.api_hash) as client:
        response = await parse_channels_by_links(client, **payload.dict())
        logger.info("Saving data to dictionary")
        df = pd.DataFrame.from_dict(response)
        df.to_json("output.json")
        return response


@app.on_event("startup")
async def main() -> None:
    logging.basicConfig(format=LOGGING_FORMAT, datefmt="%m-%d %H:%M:%S", force=True)
    logger.info("Started loading embedders")
    init_embedders()
    openai.api_key = creds.openai_api_key
