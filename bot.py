"""StudyMate Telegram bot entry point."""

import asyncio
import logging

from aiogram import Bot, Dispatcher

from config import BOT_TOKEN, validate_config
from handlers import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


async def main() -> None:
    validate_config()
    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    logging.info("StudyMate bot ishga tushdi")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("StudyMate bot to'xtatildi")
