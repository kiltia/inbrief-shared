import json
import logging
from concurrent.futures._base import TimeoutError
from datetime import datetime
from typing import List

from embedders import EmbeddingProvider, get_embedders
from telethon import TelegramClient
from telethon.errors.rpcbaseerrors import BadRequestError
from telethon.errors.rpcerrorlist import MsgIdInvalidError
from telethon.tl.functions.chatlists import CheckChatlistInviteRequest

from shared.entities import Folder, Source
from shared.utils import DATE_FORMAT

logger = logging.getLogger("app")


def get_worker(
    channel_entity,
    client: TelegramClient,
    embedders: List[EmbeddingProvider],
    social: bool,
    **kwargs,
):
    async def get_content(message) -> Source | None:
        if message.message in ["", None]:
            return None
        logger.debug(f"Started getting content for {message.id}")
        logger.debug("Parsing data from Telegram")
        content = {
            "source_id": message.id,
            "text": message.message,
            "date": message.date.strftime(DATE_FORMAT),
            "reference": f"t.me/{channel_entity.username}/{message.id}",
            "channel_id": channel_entity.id,
            "embeddings": {},
        }
        if social:
            comments = []
            try:
                async for msg in client.iter_messages(
                    channel_entity, reply_to=content["id"]
                ):
                    text = msg.text
                    if text is not None:
                        comments.append(text)
            except MsgIdInvalidError:
                logger.warn(
                    "Got invalid message ID while parsing, skipping..."
                )
            content.update({"comments": comments})
            if message.reactions is None:
                content.update({"reactions": []})
            else:
                content.update(
                    {
                        "reactions": [
                            {
                                "emoticon": reaction.reaction.emoticon,
                                "count": reaction.count,
                            }
                            for reaction in message.reactions.results
                        ]
                    }
                )

            content["reactions"] = json.dumps(content["reactions"])

        logger.debug("Started generating embeddings for text")
        # TODO(nrydanov): Move embedding retrieval out of this function
        # to enable batch processing on GPU to increase overall performance
        for emb in embedders:
            content["embeddings"].update(
                {emb.get_label(): emb.get_embeddings([message.message])[0]}
            )

        content["embeddings"] = json.dumps(content["embeddings"])
        logger.debug(f"Ended parsing {message.id}")
        return Source.parse_obj(content)

    return get_content


async def get_content_from_channel(
    channel_entity,
    client: TelegramClient,
    embedders: List[EmbeddingProvider],
    end_date,
    offset_date=None,
    **kwargs,
) -> List[Source]:
    batch: List[Source] = []
    end_date = datetime.strptime(end_date, DATE_FORMAT)
    offset_date = (
        datetime.strptime(offset_date, DATE_FORMAT) if offset_date else None
    )

    api_iterator = client.iter_messages(
        channel_entity, offset_date=offset_date
    )
    get_content = get_worker(channel_entity, client, embedders, **kwargs)
    async for message in api_iterator:
        try:
            if message.date.replace(tzinfo=None) < end_date:
                break
            source = await get_content(message)
            if source is not None:
                batch.append(source)
        except TimeoutError:
            logging.error("Received timeout when processing chunk, skipping.")
            continue

    return batch


async def retrieve_channels(ctx, chat_folder_link: str):
    logger.debug(f"Retrieving channels from link: {chat_folder_link}")

    slug = chat_folder_link.split("/")[-1]
    try:
        channels = (await ctx.client(CheckChatlistInviteRequest(slug))).chats
        ids = list(map(lambda x: x.id, channels))
        entity = Folder(chat_folder_link=chat_folder_link, channels=ids)

        await ctx.folder_repository.add_or_update(entity, ["channels"])
    except BadRequestError:
        entity = await ctx.folder_repository.get(
            "chat_folder_link", chat_folder_link
        )
        ids = entity.channels
    return ids


async def parse_channels(
    ctx,
    channels: List[int],
    required_embedders: List[str],
    **parse_args,
) -> List[Source]:
    logger.debug("Getting all required embedders")

    client = ctx.client
    embedders = get_embedders(required_embedders)
    result: List[Source] = []
    for channel_id in channels:
        channel_entity = await client.get_entity(channel_id)
        logger.debug(f"Parsing channel: {channel_entity.id}")
        response = await get_content_from_channel(
            channel_entity, client, embedders, **parse_args
        )
        result = result + response

    return result
