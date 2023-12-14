import json
import logging
from logging.config import dictConfig

from shared.utils import SHARED_CONFIG_PATH

LOGGING_FORMAT = (
    "[%(levelname)s] [%(asctime)s] %(message)s (%(correlation_id)s)"
)


def configure_logging() -> None:
    from functools import partial, partialmethod

    with open(f"{SHARED_CONFIG_PATH}/settings.json") as f:
        d = json.load(f)
        dictConfig(d["logger"])

    logging.TRACE = logging.DEBUG - 5
    logging.addLevelName(logging.TRACE, "TRACE")
    logging.Logger.trace = partialmethod(logging.Logger.log, logging.TRACE)
    logging.trace = partial(logging.log, logging.TRACE)
