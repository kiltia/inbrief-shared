import asyncio
import json
import logging
from concurrent.futures._base import TimeoutError
from datetime import datetime
from typing import List

from embedders import EmbeddingProvider, OpenAiEmbedder, get_embedders
from telethon import TelegramClient
from telethon.errors.rpcbaseerrors import BadRequestError
from telethon.errors.rpcerrorlist import MsgIdInvalidError
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.functions.chatlists import CheckChatlistInviteRequest

from shared.entities import Channel, Folder, Source
from shared.utils import DATE_FORMAT, DB_DATE_FORMAT

logger = logging.getLogger("app")


def get_worker(
    channel_entity,
    client: TelegramClient,
    openai_client,
    embedders: List[EmbeddingProvider],
    classifier,
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
            "date": message.date.strftime(DB_DATE_FORMAT),
            "reference": f"t.me/{channel_entity.username}/{message.id}",
            "channel_id": channel_entity.id,
            "embeddings": {},
            "views": message.views,
        }
        if social:
            logger.debug(f"Started getting social content for {message.id}")
            comments = []
            try:
                async for msg in client.iter_messages(
                    channel_entity, reply_to=message.id
                ):
                    text = msg.text
                    if text is not None:
                        comments.append(text)
            except MsgIdInvalidError:
                logger.warn(
                    "Got invalid message ID while parsing, skipping..."
                )
            except ValueError as e:
                logger.warn(e)
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
            logger.debug(f"Ended getting social content for {message.id}")

            content["reactions"] = json.dumps(content["reactions"])

        logger.debug(f"Started generating embeddings for {message.id}")
        # TODO(nrydanov): Move embedding retrieval out of this function
        # to enable batch processing on GPU to increase overall performance
        for emb in embedders:
            if isinstance(emb, OpenAiEmbedder):
                embeddings = (await emb.aget_embeddings([message.message], openai_client))[0]
            else:
                embeddings = emb.get_embeddings([message.message])[0]
            content["embeddings"].update({emb.get_label(): embeddings})

        content["embeddings"] = json.dumps(content["embeddings"])
        logger.debug(f"Ended generating embeddings for {message.id}")

        logger.debug(f"Started classification {message.id}")
        content.update(
            {
                "label": None
                if classifier is None
                else classifier.get_labels([message.message])[0]
            }
        )
        logger.debug(f"Ended classification {message.id}")
        logger.debug(f"Ended parsing message {message.id}")
        return Source.parse_obj(content)

    return get_content


async def get_content_from_channel(
    channel_entity,
    client: TelegramClient,
    openai_client,
    embedders: List[EmbeddingProvider],
    classifier,
    end_date,
    offset_date=None,
    **kwargs,
) -> List[Source]:
    batch = []
    end_date = datetime.strptime(end_date, DATE_FORMAT)
    offset_date = (
        datetime.strptime(offset_date, DATE_FORMAT) if offset_date else None
    )
    api_iterator = client.iter_messages(
        channel_entity, offset_date=offset_date
    )
    get_content = get_worker(
        channel_entity, client, openai_client, embedders, classifier, **kwargs
    )
    async for message in api_iterator:
        try:
            if message.date.replace(tzinfo=None) < end_date:
                break
            batch.append(get_content(message))
        except TimeoutError:
            logging.error("Received timeout when processing chunk, skipping.")
            continue

    logger.debug(f"Got {len(batch)} messages in total. Processing...")

    batch = await asyncio.gather(*batch)
    return list(filter(lambda x: x is not None, batch))


async def retrieve_channels(ctx, chat_folder_link: str):
    logger.debug(f"Retrieving channels from link: {chat_folder_link}")

    slug = chat_folder_link.split("/")[-1]
    try:
        channels = (await ctx.client(CheckChatlistInviteRequest(slug))).chats
        ids = list(map(lambda x: x.id, channels))
        entity = Folder(chat_folder_link=chat_folder_link, channels=ids)

        await ctx.folder_repository.add_or_update(entity, ["channels"])
        logger.info(
            f"Successful update channels from link: {chat_folder_link}"
        )
    except BadRequestError:
        logger.info(
            f"Warning when updating channels from link: {chat_folder_link}"
        )
        entity = (
            await ctx.folder_repository.get(
                "chat_folder_link", chat_folder_link
            )
        )[0]
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
    openai_client = ctx.openai_client
    embedders = get_embedders(required_embedders)
    classifier = ctx.classifier
    result: List[Source] = []
    for channel_id in channels:
        channel_entity = await client.get_entity(channel_id)
        logger.debug(f"Parsing channel: {channel_entity.id}")
        info = (await client(GetFullChannelRequest(channel_id))).full_chat
        channel = Channel(
            channel_id=info.id,
            title=channel_entity.title,
            about=info.about,
            subscribers=info.participants_count,
        )
        await ctx.channel_repository.add_or_update(
            channel, fields=["title", "about", "subscribers"]
        )
        response = await get_content_from_channel(
            channel_entity, client, openai_client, embedders, classifier, **parse_args
        )
        result = result + response

    return result
