import logging
from argparse import ArgumentParser

import openai
import pandas as pd
from config import Credentials, ParserSettings
from telethon.sync import TelegramClient
from utils import LOGGING_FORMAT

from scraper import parse_channels_by_links

logger = logging.getLogger(__name__)


def init_parser() -> ArgumentParser:
    logging.info("Initializing parser")
    parser = ArgumentParser()
    parser.add_argument("-o", type=str, required=True, dest="output")
    return parser


def main() -> None:
    logging.basicConfig(format=LOGGING_FORMAT, datefmt="%m-%d %H:%M:%S", force=True)
    creds = Credentials()
    openai.api_key = creds.openai_api_key
    parser = init_parser()
    parser.parse_args()

    with TelegramClient(creds.phone, creds.api_id, creds.api_hash) as client:
        parser_config = ParserSettings()
        ans = parse_channels_by_links(client, **parser_config.dict())
        df = pd.DataFrame.from_dict(ans)
        print(df.head())
        df.to_json(parser.output, orient="records")


if __name__ == "__main__":
    main()
