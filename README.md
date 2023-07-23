# Scraper

This module is responsible for information retrieval — text data, embeddings,
social features, etc.

To work with it you need to create Telegram application, add `api_id`, `api_hash`,
`phone` fields in .env file and place it to config folder. 

NOTE: if you want to use embeddings from OpenAI API, you also need to pass
`openai_api_key` from paid account.

## Running

This application is built using FastAPI, so you may use
```
uvicorn --app-dir=src main:app --reload
```
command.
