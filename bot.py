"""
bot.py — StudyMate botning asosiy fayli.
Barcha handler'larni ulaydi va botni ishga tushiradi.
"""
import asyncio
import logging
from aiogram import Bot, Dispatcher

from config import BOT_TOKEN
from handlers import router

# Log sozlamalari
logging.basicConfig(level=logging.INFO)


async def main():
    # Bot va Dispatcher ni yaratamiz
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()

    # Handler'larni ulaymiz (handlers.py dan)
    dp.include_router(router)

    print("🚀 StudyMate bot ishga tushdi!")
    # Polling — yangi xabarlarni kutish
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
