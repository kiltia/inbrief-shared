import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import TimeoutError
from datetime import datetime

import openai_api
from telethon.errors.rpcerrorlist import MsgIdInvalidError
from tqdm import tqdm
from utils import DATE_FORMAT, DEFAULT_END_DATE, PAYLOAD_STRUCTURE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_worker(channel_entity, client, social, markup, categories, model, max_retries):
    def get_content(message):
        content = {
            "id": [message.id],
            "text": [message.message],
            "date": [message.date.strftime(DEFAULT_END_DATE)],
            "channel": [channel_entity.id],
        }
        if social:
            comments = []
            try:
                for msg in client.iter_messages(channel_entity, reply_to=content["id"]):
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
        else:
            content["reactions"] = [None]
            content["comments"] = [None]
        if markup:
            content["cls"] = [
                openai_api.classify(message.message, categories, model, max_retries)
            ]
        return content

    return get_content


def merge_payloads(collected, current):
    return {k: v1 + v2 for (k, v1), (k, v2) in zip(collected.items(), current.items())}


def get_chunked_messages(it, size, end_date):
    while True:
        batch = []
        stop = False
        for _ in range(size):
            try:
                msg = next(it)
                if msg.date.replace(tzinfo=None) > end_date:
                    batch.append(next(it))
                else:
                    stop = True
            except StopIteration:
                stop = True
        yield batch

        if stop:
            break


def get_content_from_channel(
    channel_entity,
    client,
    offset_date=None,
    end_date=DEFAULT_END_DATE,
    num_workers=8,
    **kwargs,
):
    batch = PAYLOAD_STRUCTURE.copy()
    end_date = datetime.strptime(end_date, DATE_FORMAT)
    offset_date = datetime.strptime(offset_date, DATE_FORMAT) if offset_date else None

    api_iterator = client.iter_messages(channel_entity, offset_date=offset_date)
    get_content = get_worker(channel_entity, client, **kwargs)

    for chunk in tqdm(get_chunked_messages(api_iterator, num_workers, end_date)):
        with ThreadPoolExecutor(num_workers) as executor:
            try:
                # NOTE(nrydanov): Probably we need to find more appropriate timeout
                processed_chunk = list(executor.map(get_content, chunk, timeout=30))
            except TimeoutError:
                logging.error("Received timeout when processing chunk, skipping.")
                continue
            for processed in processed_chunk:
                batch = merge_payloads(batch, processed)
    return batch


def parse_channels_by_links(client, channels, **parse_args):
    payload = PAYLOAD_STRUCTURE.copy()

    for channel in channels:
        channel_entity = client.get_entity(channel)
        print(f"Parsing {channel_entity.title} ...")
        response = get_content_from_channel(channel_entity, client, **parse_args)
        payload = merge_payloads(payload, response)

    return payload


def get_all_clients_channels(client):
    channels = []
    for dialog in client.iter_dialogs():
        if not dialog.is_group and dialog.is_channel:
            channels.append(dialog.entity)
    return channels


def parse_client_channels(client, **parse_args):
    result = {
        "channel": [],
        "id": [],
        "text": [],
        "date": [],
        "comments": [],
        "reactions": [],
    }
    channels_entity = get_all_clients_channels(client)
    for channel_entity in channels_entity:
        print(f"Parsing {channel_entity.title} ...")
        get_content_from_channel(channel_entity, client, result, **parse_args)
    return result
