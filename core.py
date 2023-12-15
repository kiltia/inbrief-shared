import logging

import numpy as np
import openai
from tenacity import (
    after_log,
    before_log,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from openai_api.prompts import (
    CLASSIFY_TASK,
    get_editor_context,
    get_summary_context,
    get_title_context,
)

from .utils import count_tokens

logger = logging.getLogger("app")


base_retry = retry(
    retry=retry_if_exception(openai.error.RateLimitError),
    wait=wait_exponential(min=2, max=30, multiplier=1.5),
    after=after_log(logger, log_level=logging.DEBUG),
    before=before_log(logger, log_level=logging.DEBUG),
    before_sleep=before_sleep_log(logger, logging.DEBUG),
    reraise=True,
    stop=stop_after_attempt(5),
)


@base_retry
async def aget_embeddings(input, model):
    embs = (await openai.Embedding.acreate(input=input, model=model))["data"]

    return list(map(lambda x: x["embedding"], embs))


@base_retry
def get_embeddings(input, model):
    embs = openai.Embedding.create(input=input, model=model)["data"]

    return list(map(lambda x: x["embedding"], embs))


@base_retry
def summarize(
    input,
    model,
    max_tokens=400,
    temperature=0.1,
    presense_penalty=0,
    timeout=30,
    additional_context=None,
):
    messages = get_summary_context(input, max_tokens)
    if additional_context is not None:
        messages.append({"role": "system", "content": additional_context})
    logger.debug(
        f"Sending summary request to OpenAI with {count_tokens(messages, model)} tokens"
    )
    return (
        openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            presence_penalty=presense_penalty,
            timeout=timeout,
            max_tokens=max_tokens * 2,
        )
    )["choices"][0]["message"]["content"]


# Outdated, need to make it equal to synchronous version when needed
@base_retry
async def asummarize(
    input,
    model,
    max_tokens=300,
    temperature=0.2,
    presense_penalty=-1.5,
    timeout=30,
    additional_context=None,
):
    messages = get_summary_context(input, max_tokens)
    logger.debug(
        f"Sending summary request to OpenAI with {count_tokens(messages, model)} tokens"
    )
    return (
        await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            temperature=temperature,
            presence_penalty=presense_penalty,
            timeout=timeout,
            max_tokens=max_tokens * 2,
        )
    )["choices"][0]["message"]["content"]


@base_retry
def get_title(input, model, max_tokens=30):
    messages = get_title_context(input, max_tokens)
    logger.debug(
        f"Sending title request to OpenAI with {count_tokens(messages, model)} tokens"
    )
    return (
        openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.2,
            presence_penalty=-1.5,
            timeout=30,
            max_tokens=max_tokens * 2,
        )
    )["choices"][0]["message"]["content"]


@base_retry
async def aget_title(input, model, max_tokens=30):
    messages = get_title_context(input, max_tokens)
    logger.debug(
        f"Sending title request to OpenAI with {count_tokens(messages, model)} tokens"
    )
    response = await openai.ChatCompletion.acreate(
        model=model,
        messages=messages,
        temperature=0.2,
        presence_penalty=-1.5,
        timeout=30,
        max_tokens=max_tokens * 2,
    )

    logger.debug(f"Got response from OpenAI: {response}")

    return response["choices"][0]["message"]["content"]


@base_retry
def edit(
    input,
    model,
    style,
    max_tokens=800,
    temperature=0.2,
    presense_penalty=-1.5,
    timeout=30,
):
    messages = get_editor_context(input, max_tokens, style)
    logger.debug(
        f"Sending edit request to OpenAI with {count_tokens(messages, model)} tokens"
    )
    return (
        openai.ChatCompletion.create(
            model=model,
            messages=get_editor_context(input, max_tokens, style),
            temperature=temperature,
            presence_penalty=presense_penalty,
            timeout=timeout,
            max_tokens=max_tokens,
        )
    )["choices"][0]["message"]["content"]


@base_retry
async def aedit(
    input,
    model,
    style,
    max_tokens=400,
    temperature=0.2,
    presense_penalty=-1.5,
    timeout=30,
):
    messages = get_editor_context(input, max_tokens, style)
    logger.debug(
        f"Sending edit request to OpenAI with {count_tokens(messages, model)} tokens"
    )
    return (
        await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            temperature=temperature,
            presence_penalty=presense_penalty,
            timeout=timeout,
            max_tokens=max_tokens,
        )
    )["choices"][0]["message"]["content"]


def classify_attempt(attempt, categories, max_retries, **kwargs):
    def validate_response(response, categories):
        if response in categories:
            return (True, response)
        else:
            return (
                False,
                "Выводи только название одного из классов, перечисленных мной ранее,",
            )

    if attempt > max_retries:
        return None
    logger.debug(f"Creating request {attempt + 1} for chat completion")
    completion = openai.ChatCompletion.create(**kwargs)
    response = completion["choices"][0]["message"]["content"]
    logger.debug(f"Got response from OpenAI: {response}")
    status, value = validate_response(response, categories)
    if status:
        logging.debug("Response has correct format")
        return value
    else:
        messages = kwargs.pop("messages")
        messages.append(
            {
                "role": "user",
                "content": value
                + "И напоминаю, не выводи ничего, кроме одного слова — названия класса.",
            }
        )
        logging.debug("Response to model: " + messages[-1]["content"])
        logging.warn("Got bad format from model. Trying another attempt")
        return classify_attempt(
            attempt + 1,
            categories,
            messages=messages,
            max_retries=max_retries,
            **kwargs,
        )


# TODO(nrydanov): Move stop_after_attempt to configuration file
# NOTE(nrydanov): This method make paid request to OpenAI
@retry(
    wait=wait_exponential(min=2, max=60, multiplier=2),
    after=after_log(logger, logging.INFO),
    before=before_log(logger, log_level=logging.DEBUG),
    before_sleep=before_sleep_log(logger, logging.DEBUG),
    stop=stop_after_attempt(3),
    reraise=True,
)
def classify(text, categories, model, max_retries):
    class_list = f"Вот список классов: {','.join(categories)}"
    max_tokens = np.max(list(map(lambda x: len(x), categories)))
    return classify_attempt(
        0,
        categories,
        model=model,
        messages=[
            {"role": "system", "content": CLASSIFY_TASK + "\n" + class_list},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
        presence_penalty=-1.5,
        timeout=30,
        max_tokens=int(max_tokens),
        max_retries=max_retries,
    )
