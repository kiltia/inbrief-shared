import logging

from config import DefaultClusteringSettings, DefaultSearcherSettings, LinkerAPISettings
from fastapi import FastAPI
from matcher import Matcher
from models import LinkingRequest
from utils import LOGGING_FORMAT

app = FastAPI()


@app.post("/get_stories")
def get_stories(request: LinkingRequest):
    matcher = Matcher(request.texts, request.embeddings, request.dates)
    stories, stories_nums = matcher.get_stories(request.method, **request.config)

    response = {"stories": stories, "stories_nums": stories_nums}
    return response


@app.get("/")
def hello_world():
    return "Linker API is running!"


@app.on_event("startup")
async def main() -> None:
    logging.basicConfig(
        format=LOGGING_FORMAT, datefmt="%m-%d %H:%M:%S", level=logging.DEBUG, force=True
    )
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
