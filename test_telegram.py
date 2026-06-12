"""
NEXUS — Telegram Alert Test
Run this locally to verify your Telegram bot is working.

Usage:
  python test_telegram.py

Reads credentials from .env file (never hardcode them).
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID",   "")

if not TOKEN or not CHAT_ID:
    print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in your .env file")
    print("Copy .env.example to .env and fill in the values")
    exit(1)

msg = (
    "*NEXUS TEST MESSAGE*\n\n"
    "Telegram alerts are working correctly.\n\n"
    "You will receive a real alert whenever NEXUS\n"
    "generates a BUY or SELL signal.\n\n"
    "_HOLD signals do not trigger alerts._"
)

r = requests.post(
    f"https://api.telegram.org/bot{TOKEN}/sendMessage",
    json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
)

data = r.json()
if data.get("ok"):
    print("SUCCESS — check your Telegram for the message")
else:
    print(f"FAILED — {data.get('description', 'unknown error')}")
    print("Common fixes:")
    print("  - Did you send /start to your bot in Telegram?")
    print("  - Is the token correct?")
    print("  - Is the chat_id correct?")
