#!/usr/bin/env python3

from telethon import TelegramClient

from config import Credentials

creds = Credentials()

with TelegramClient(
    "sessions/dev",
    creds.api_id,
    creds.api_hash,
    system_version="4.16.30-vxCUSTOM",
) as client:
    client.session.save()
