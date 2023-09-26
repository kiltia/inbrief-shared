import logging

from fastapi import FastAPI
from matcher import Matcher

from shared.models import LinkingRequest
from shared.utils import LOGGING_FORMAT

app = FastAPI()


@app.post("/get_stories")
def get_stories(request: LinkingRequest):
    matcher = Matcher(request.texts, request.embeddings, request.dates)
    stories, stories_nums = matcher.get_stories(
        request.method, **request.config
    )

    response = {"stories": stories, "stories_nums": stories_nums}
    return response


@app.get("/")
def hello_world():
    return "Linker API is running!"


@app.on_event("startup")
async def main() -> None:
    logging.basicConfig(
        format=LOGGING_FORMAT,
        datefmt="%m-%d %H:%M:%S",
        level=logging.DEBUG,
        force=True,
    )
