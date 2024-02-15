import logging

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from matcher import Matcher

from shared.models import LinkingRequest
from shared.routes import LinkerRoutes

app = FastAPI()
app.add_middleware(CorrelationIdMiddleware, validator=None)


logger = logging.getLogger("app")


@app.post(LinkerRoutes.GET_STORIES)
async def get_stories(request: LinkingRequest):
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

    if request.return_plot_data:
        return {"results": results, "embeddings": embeddings.tolist()}
    else:
        return {"results": results}


@app.get("/")
def hello_world():
    return "Linker API is running!"
