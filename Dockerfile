FROM python:3.9

WORKDIR /linker

COPY config config
COPY src src
COPY pyproject.toml .
COPY poetry.lock .

RUN pip3 install poetry
ENV PATH="${PATH}:/root/.poetry/bin"

RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi
CMD ["python3", "src/main.py"]
