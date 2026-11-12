"""
bot.py — StudyMate botning asosiy fayli.
Barcha handler'larni ulaydi va botni ishga tushiradi.
"""
import asyncio
import logging

from aiogram import Bot, Dispatcher

from config import BOT_TOKEN, GROQ_API_KEY
from database import init_db
from handlers import router

# Log sozlamalari
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    missing = [
        name for name, value in (("BOT_TOKEN", BOT_TOKEN), ("GROQ_API_KEY", GROQ_API_KEY))
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing environment variables: {', '.join(missing)} (see .env.example)")

    init_db()

    # Bot va Dispatcher ni yaratamiz
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()

    # Handler'larni ulaymiz (handlers.py dan)
    dp.include_router(router)

    logger.info("StudyMate bot started")
    # Polling — yangi xabarlarni kutish
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
