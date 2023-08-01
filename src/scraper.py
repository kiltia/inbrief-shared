import copy
import logging
from concurrent.futures._base import TimeoutError
from datetime import datetime

import openai_api
from embedders import get_embedders
from telethon.errors.rpcerrorlist import MsgIdInvalidError
from utils import (
    DATE_FORMAT,
    DEFAULT_END_DATE,
    DEFAULT_PAYLOAD_STRUCTURE,
    add_optional_columns,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_worker(channel_entity, client, embedders, social, markup, **kwargs):
    async def get_content(message):
        content = {
            "id": [message.id],
            "text": [message.message],
            "date": [message.date.strftime(DATE_FORMAT)],
            "channel": [channel_entity.id],
        }

        if social:
            comments = []
            try:
                async for msg in client.iter_messages(
                    channel_entity, reply_to=content["id"]
                ):
                    text = msg.text
                    if not (text is None):
                        comments.append(text)
            except MsgIdInvalidError:
                comments = None
            content["comments"] = [comments]
            if message.reactions is None:
                content["reactions"] = [None]
            else:
                content["reactions"] = [
                    [
                        (reaction.reaction.emoticon, reaction.count)
                        for reaction in message.reactions.results
                    ]
                ]

        if markup:
            classify_args = ["categories", "classify_model", "max_retries"]
            args = [kwargs.pop(x) for x in classify_args]
            if message.message == "":
                content["cls"] = [None]
            else:
                content["cls"] = [openai_api.classify(message.message, *args)]

        # TODO(nrydanov): Move embedding retrieval out of this function
        # to enable batch processing on GPU to increase overall performance
        for emb in embedders:
            if not message.message or message.message == "":
                content[emb.get_label()] = [None]
            else:
                content[emb.get_label()] = [emb.get_embeddings([message.message])[0]]

        return content

    return get_content


def merge_payloads(collected, current):
    for key in collected.keys():
        collected[key] += current[key]


async def get_content_from_channel(
    channel_entity,
    client,
    embedders,
    scheme,
    offset_date=None,
    end_date=DEFAULT_END_DATE,
    **kwargs,
):
    batch = copy.deepcopy(scheme)
    end_date = datetime.strptime(end_date, DATE_FORMAT)
    offset_date = datetime.strptime(offset_date, DATE_FORMAT) if offset_date else None

    api_iterator = client.iter_messages(channel_entity, offset_date=offset_date)
    get_content = get_worker(channel_entity, client, embedders, **kwargs)
    async for message in api_iterator:
        try:
            if message.date.replace(tzinfo=None) < end_date:
                break
            response = await get_content(message)
            merge_payloads(batch, response)
        except TimeoutError:
            logging.error("Received timeout when processing chunk, skipping.")
            continue

    return batch


async def parse_channels_by_links(client, channels, required_embedders, **parse_args):
    logger.info("Getting all required embedders")
    embedders = get_embedders(required_embedders)
    scheme = add_optional_columns(
        DEFAULT_PAYLOAD_STRUCTURE, embedders, parse_args["markup"], parse_args["social"]
    )

    payload = copy.deepcopy(scheme)
    for channel in channels:
        logger.info(f"Parsing channel ({channel})")
        channel_entity = await client.get_entity(channel)
        response = await get_content_from_channel(
            channel_entity, client, embedders, scheme, **parse_args
        )
        merge_payloads(payload, response)

    return payload
