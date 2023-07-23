FROM python:3.11 AS builder

ENV PIP_DEFAULT_TIMEOUT=200 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.5.1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

ENV WD_NAME=/scraper
ARG PRIVATE_KEY_PATH

WORKDIR $WD_NAME

COPY poetry.lock pyproject.toml .
ADD $PRIVATE_KEY_PATH /root/.ssh/id_ed25519
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN pip install poetry==${POETRY_VERSION}
RUN poetry config installer.max-workers 10 \
        && poetry install --no-dev --no-interaction --no-ansi -vvv

FROM python:3.11-slim as runtime

ENV WD_NAME=/scraper
WORKDIR $WD_NAME

ENV PATH="$WD_NAME/.venv/bin/:$PATH"

COPY --from=builder $WD_NAME/.venv .venv
COPY src src
CMD ["uvicorn", "--app-dir", "src", "--host", "0.0.0.0", "main:app"]
