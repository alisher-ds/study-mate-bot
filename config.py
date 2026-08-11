"""
config.py — Botning barcha sozlamalari.
Tokenlar .env faylidan o'qiladi (xavfsizlik uchun kodga yozilmaydi).
"""
import os
from dotenv import load_dotenv

# .env fayldan o'zgaruvchilarni yuklaymiz
load_dotenv()

# Telegram bot tokeni (BotFather dan olinadi)
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Groq API kaliti (AI javoblar uchun) — groq.com dan olinadi
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Ishlatiladigan AI modeli
AI_MODEL = "llama-3.3-70b-versatile"

# Foydalanuvchi bazasi fayli nomi
DB_NAME = "users.db"
