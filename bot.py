from dotenv import load_dotenv

# .env faylni yuklash - BU ENG BIRINCHI AMALGA OSHIRILISHI KERAK
load_dotenv()

import asyncio
import os

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

# pdf_processor.py dan funksiyalarni import qilamiz
from pdf_processor import extract_text_from_pdf, split_into_chunks

# rag_engine.py dan funksiyalarni import qilamiz
from rag_engine import add_document, search_relevant_chunks, generate_answer, generate_quiz, generate_summary

# Bot va Dispatcher yaratamiz
bot = Bot(token=os.getenv("BOT_TOKEN"))
dp = Dispatcher(storage=MemoryStorage())
router = Router()
dp.include_router(router)


# Foydalanuvchi holatlari uchun StatesGroup klassi
class UserState(StatesGroup):
    waiting_question = State()  # Savol kutish holati


def main_menu() -> ReplyKeyboardMarkup:
    """
    Asosiy menyu tugmalarini yaratadi.
    Tugmalar 2 qatorda joylashgan.
    """
    keyboard = [
        [KeyboardButton(text="📄 Fayl yuklash"), KeyboardButton(text="❓ Savol berish")],
        [KeyboardButton(text="📝 Test tuzish"), KeyboardButton(text="📌 Xulosa")]
    ]
    return ReplyKeyboardMarkup(keyboard=keyboard, resize_keyboard=True)


@router.message(Command("start"))
async def cmd_start(message: Message):
    """
    /start buyrug'i uchun handler.
    Foydalanuvchini salomlaydi va asosiy menyuni ko'rsatadi.
    """
    await message.answer(
        "Assalomu alaykum! StudyMate botiga xush kelibsiz.\n\n"
        "Men sizning PDF fayllaringiz asosida savollaringizga javob beraman, "
        "testlar tuzaman va matnlarni xulosalayman.\n\n"
        "Quyidagi tugmalardan birini tanlang:",
        reply_markup=main_menu()
    )


@router.message(F.document)
async def handle_document(message: Message):
    """
    PDF fayl qabul qiluvchi handler.
    Faylni yuklab oladi, matnni ajratadi va bazaga saqlaydi.
    """
    # Fayl PDF ekanligini tekshiramiz (ixtiyoriy, lekin tavsiya etiladi)
    if not message.document.file_name.endswith('.pdf'):
        await message.answer("Iltimos, faqat PDF fayl yuboring.")
        return

    # Foydalanuvchiga xabar beramiz
    processing_msg = await message.answer("Faylni qayta ishlayapman...")

    try:
        # Faylni yuklab olamiz
        file_info = await bot.get_file(message.document.file_id)
        temp_file_path = f"temp_{message.from_user.id}_{message.document.file_name}"
        
        await bot.download_file(file_info.file_path, temp_file_path)

        # PDF dan matnni ajratib olamiz
        full_text = extract_text_from_pdf(temp_file_path)

        # Matnni kichik qismlarga (chunk) bo'lamiz
        chunks = split_into_chunks(full_text)

        # Bazaga saqlaymiz
        doc_name = message.document.file_name
        add_document(user_id=message.from_user.id, chunks=chunks, doc_name=doc_name)

        # Vaqtinchalik faylni o'chiramiz
        os.remove(temp_file_path)

        await processing_msg.edit_text(
            f"Fayl muvaffaqiyatli qayta ishlandi!\n"
            f"Matn {len(chunks)} ta qismga ajratildi va bazaga saqlandi."
        )
    except Exception as e:
        await processing_msg.edit_text(f"Faylni qayta ishlashda xatolik yuz berdi: {str(e)}")
        # Xatolik bo'lsa ham faylni o'chirishga harakat qilamiz
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.message(F.text == "❓ Savol berish")
async def ask_question_button(message: Message, state: FSMContext):
    """
    'Savol berish' tugmasi bosilganda ishlaydi.
    Foydalanuvchidan savol so'raydi va holatni o'zgartiradi.
    """
    await message.answer("Savolingizni yozing:")
    await state.set_state(UserState.waiting_question)


@router.message(UserState.waiting_question)
async def process_question(message: Message, state: FSMContext):
    """
    Foydalanuvchi savol yozganda ishlaydi.
    Mos keluvchi ma'lumotlarni topadi va AI orqali javob generatsiya qiladi.
    """
    query = message.text
    
    # Avval mos keluvchi qismlarni qidiramiz
    relevant_chunks = search_relevant_chunks(user_id=message.from_user.id, query=query)
    
    # Agar hech narsa topilmasa
    if not relevant_chunks:
        await message.answer(
            "Kechirasiz, hozircha bu savolga javob berish uchun yetarli ma'lumot topilmadi.\n"
            "Iltimos, avval tegishli PDF faylni yuklang."
        )
        await state.clear()
        return

    # Javob generatsiya qilamiz
    await message.answer("Javob tayyorlanmoqda...")
    answer = generate_answer(query=query, relevant_chunks=relevant_chunks)
    
    await message.answer(answer)
    
    # Holatni tozalaymiz
    await state.clear()


@router.message(F.text == "📝 Test tuzish")
async def create_quiz(message: Message):
    """
    'Test tuzish' tugmasi bosilganda ishlaydi.
    Yuklangan fayllar asosida test savollari generatsiya qiladi.
    """
    await message.answer("Test savollari tayyorlanmoqda...")
    
    quiz = generate_quiz(user_id=message.from_user.id, num_questions=5)
    
    await message.answer(quiz)


@router.message(F.text == "📌 Xulosa")
async def create_summary(message: Message):
    """
    'Xulosa' tugmasi bosilganda ishlaydi.
    Yuklangan fayllar asosida qisqa xulosa tayyorlaydi.
    """
    await message.answer("Xulosa tayyorlanmoqda...")
    
    summary = generate_summary(user_id=message.from_user.id)
    
    await message.answer(summary)


@router.message(F.text == "📄 Fayl yuklash")
async def upload_file_instruction(message: Message):
    """
    'Fayl yuklash' tugmasi bosilganda ishlaydi.
    Foydalanuvchiga PDF fayl yuborishni taklif qiladi.
    """
    await message.answer("Iltimos, PDF faylni yuboring.")


async def main():
    """
    Botni ishga tushiradi.
    """
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
