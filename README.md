# Inbrief
Let us tell you in brief about Inbrief

## Description

InBrief aims to create a tool for healthier and more mindful consumption of information streams, focusing initially on Telegram channels. The project's goal is to facilitate the aggregation and summarization of news from a selected set of channels, offering users a more digestible and relevant content experience. 

Future developments will introduce advanced filtering and tagging options to enable users to access only what is essential, minimizing exposure to irrelevant information. This could be achieved through manual settings or automated learning mechanisms, leveraging like/dislike buttons to tailor content feeds. Inbrief project serves as a digital curator, streamlining the vast influx of daily news into tailored, meaningful insights.

## Components 

Open-source side of this project contains several microservices that perform crawling and news grouping:
1. **Scraper** provides API for scraping range of posts from one date to another via Telethon
1. **Linker** is a service that groups separate posts together, forming, so-called, story
1. There's also a **Dashboard** service, which provides useful interface for experiments

All components except Telegram bot exist in their own folder at top-level.

## Configuration & Dependencies

### Global environment configuration

#### Database credentials

Postgres uses `POSTGRES_DB`, `POSTGRES_USER` and `POSTGRES_PASSWORD` to start database with correct credentials. 

**NOTE:** you should also use same password in global `config/settings.json` file. 

#### Docker Compose configuration

As there're two compose files for dev/prod environment, you can add `COMPOSE_FILE` fields with `docker-compose.[dev\prod].yaml` value. 
It will allow you to run Docker Compose commands without adding `-f` explicitly.

#### OpenAI configuration

You should add `OPENAI_API_KEY` field to let Scraper and Summarizer use OpenAI for their requests. This key should be generated for each developer separately [here](https://api.openai.com).

#### Example

```dotenv
POSTGRES_DB=inbrief
POSTGRES_USER=inbrief
POSTGRES_PASSWORD=1234
OPENAI_API_KEY=sk-AOplJoBT2W4ja11MbW02343BlbkAAGdHUKN123DWQFvhqLavA
COMPOSE_FILE=docker-compose.dev.yaml

# Optional
LINKER_WORKERS=...
# and other...
```

### Scraper

Scraper requires `.env` file in service root folder. There're some environment variables that are required for service to work: `api_id`, `api_hash` and `session`.

You can retrieve `api_id` and `api_hash` after creating application using this [guide](https://core.telegram.org/api/obtaining_api_id).

Scraper uses Telethon library to get access for Telegram API. 

To make it easier, we've created `scraper/src/generate_session.py` script, which may be called via `just gen_session` shortcut. 
For now, this script generates string session, which should be saved in `session` value.

Example:
```dotenv
api_id=11111111
api_hash=fa60db70daf43b68efe92faa8ade75fa
session=dev
```

### Linker

These services don't have any required pre-configuration themselves.


### Dependencies

You also need to add model weights for in-house models to make them work. If you don't know where to find them, please contact @nrydanov or @MakDaffi.

## How to start?

We use Rye for dependencies, so you'll need to install him before continue.

To start, you need to run `just init` in project folder. It will download dependencies specified by `pyproject.toml`, create session (if it wasn't created yet), and build images for all services.

Then, you can use `docker compose up`.

**NOTE:** You need VPN in one of allowed countries to use OpenAI API.

### Development

You can use almost all services inside containers and develop them with hot-reloading. 
Just use `docker-compose.dev.yaml` and add `FLAGS=--reload` before `docker-compose up` in your terminal.

You can also customize `components` section in `config/settings.json` file to disable some heavy parts of application.
