FROM python:3.11 AS builder

ENV PIP_DEFAULT_TIMEOUT=200 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 

ENV RYE_HOME="/opt/rye"
ENV PATH="$RYE_HOME/shims:$PATH"
ENV UV_HTTP_TIMEOUT=1200
ENV WD_NAME=/app
WORKDIR $WD_NAME
RUN curl -sSf https://rye-up.com/get | RYE_INSTALL_OPTION="--yes" \
                                       RYE_NO_AUTO_INSTALL=1  \
                                       RYE_TOOLCHAIN_VERSION="3.11" \
                                       bash \
&& rye config --set-bool behavior.use-uv=true


RUN \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=requirements.lock,target=requirements.lock \
    --mount=type=bind,source=requirements-dev.lock,target=requirements-dev.lock \
    --mount=type=bind,source=.python-version,target=.python-version \
    rye sync --no-lock --no-dev \
    && rye run python -W ignore -m nltk.downloader -d .venv/nltk_data punkt stopwords


ENV PATH="$WD_NAME/.venv/bin:$PATH"
COPY openai_api openai_api
RUN pip install -e openai_api
COPY shared shared
RUN pip install -e shared

FROM python:3.11-slim as runtime

ENV WD_NAME=/app
WORKDIR $WD_NAME

ENV PATH="$WD_NAME/.venv/bin:$PATH"
ENV PYTHONPATH="$PYTHONPATH:$WD_NAME/.venv/lib/python3.11/site-packages"

COPY --from=builder /opt/rye /opt/rye
COPY --from=builder $WD_NAME/.venv .venv
COPY --from=builder $WD_NAME/shared shared
COPY --from=builder $WD_NAME/openai_api openai_api
COPY src src
ENTRYPOINT ["uvicorn", "--app-dir", "src", "--host", "0.0.0.0", "main:app"]
