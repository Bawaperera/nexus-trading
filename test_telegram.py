import requests

TOKEN   = "8304315151:AAFHM7Jm5fQYjMkfS5yp10hAQWoDQjR7HIY"     # from BotFather
CHAT_ID = 6815273629       # from @userinfobot

msg = (
    "*NEXUS TEST MESSAGE*\n\n"
    "If you see this, Telegram alerts are working.\n\n"
    "You will get a real alert whenever NEXUS\n"
    "generates a BUY or SELL signal."
)

r = requests.post(
    f"https://api.telegram.org/bot{TOKEN}/sendMessage",
    json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
)

print("Status:", r.status_code)
print("Response:", r.json())