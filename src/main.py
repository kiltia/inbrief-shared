import logging
from contextlib import asynccontextmanager

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from matcher import Matcher
from pydantic import TypeAdapter

from shared.logger import configure_logging
from shared.models import LinkingRequest, LinkingResponse
from shared.routes import LinkerRoutes

app = FastAPI()
app.add_middleware(CorrelationIdMiddleware, validator=None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    yield


logger = logging.getLogger("app")


@app.post(LinkerRoutes.GET_STORIES)
async def get_stories(request: LinkingRequest) -> LinkingResponse:
    logger.info(f"Method: {request.config.method}")
    matcher = Matcher(
        request.entries,
        request.config.embedding_source,
        request.config.scorer,
        request.config.metric,
    )

    results, embeddings = matcher.get_stories(
        method_name=request.config.method,
        **request.settings,
        return_plot_data=request.return_plot_data,
    )

    adapter = TypeAdapter(LinkingResponse)
    if request.return_plot_data:
        return adapter.validate_python(
            {"results": results, "embeddings": embeddings.tolist()}
        )
    else:
        return adapter.validate_python({"results": results})


@app.get("/")
def hello_world():
    return "Linker API is running!"
