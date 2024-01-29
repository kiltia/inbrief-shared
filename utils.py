import tiktoken
from openai import AsyncOpenAI

CLASSIFY_PATTERN = r"\d{1}"
LOGGING_FORMAT = "[%(levelname)s] [%(asctime)s] %(message)s"


def count_tokens(messages, model):
    cnt = 0
    enc = tiktoken.encoding_for_model(model)

    for message in messages:
        cnt += len(enc.encode(message["content"]))
    return cnt


def get_async_client(api_key):
    client = AsyncOpenAI(
        api_key=api_key
    )
    return client
