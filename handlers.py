import asyncio
import logging
import os
import tempfile
from pathlib import Path

from aiogram import F, Router, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove

from config import MAX_PDF_SIZE_MB
from database import add_user, get_user
from pdf_processor import extract_text_from_pdf, split_into_chunks
from rag_engine import (
    add_document,
    generate_answer,
    generate_quiz,
    generate_summary,
    search_relevant_chunks,
)

router = Router()
logger = logging.getLogger(__name__)


class Registration(StatesGroup):
    ism = State()
    telefon = State()
    shahar = State()


phone_keyboard = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="📱 Telefonimni jo'natish", request_contact=True)]],
    resize_keyboard=True,
    one_time_keyboard=True,
)


@router.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    existing_user = get_user(message.from_user.id)
    if existing_user:
        await message.answer(f"Assalomu alaykum, {existing_user[1]}! 👋\n\nPDF yuboring yoki savolingizni yozing.")
        return
    await message.answer("Assalomu alaykum! 🌟 StudyMate'ga xush kelibsiz!\n\nAvval ismingizni yozing:")
    await state.set_state(Registration.ism)


@router.message(Registration.ism, F.text)
async def process_ism(message: types.Message, state: FSMContext):
    name = message.text.strip()
    if not name or len(name) > 100:
        await message.answer("Iltimos, ismingizni to'g'ri kiriting.")
        return
    await state.update_data(ism=name)
    await message.answer("Endi telefon raqamingizni yuboring:", reply_markup=phone_keyboard)
    await state.set_state(Registration.telefon)


@router.message(Registration.telefon, F.contact)
async def process_telefon(message: types.Message, state: FSMContext):
    if message.contact.user_id and message.contact.user_id != message.from_user.id:
        await message.answer("Iltimos, o'zingizning telefon raqamingizni yuboring.")
        return
    await state.update_data(telefon=message.contact.phone_number)
    await message.answer("Rahmat! 📱 Endi shahringizni yozing:", reply_markup=ReplyKeyboardRemove())
    await state.set_state(Registration.shahar)


@router.message(Registration.shahar, F.text)
async def process_shahar(message: types.Message, state: FSMContext):
    city = message.text.strip()
    if not city or len(city) > 100:
        await message.answer("Iltimos, shahar nomini to'g'ri kiriting.")
        return
    data = await state.get_data()
    add_user(message.from_user.id, data["ism"], data["telefon"], city)
    await state.clear()
    await message.answer("Barakalloh! 🎉 Ro'yxatdan o'tdingiz.\n\nPDF yuboring yoki savolingizni yozing.")


@router.message(F.document)
async def handle_pdf(message: types.Message):
    doc = message.document
    filename = Path(doc.file_name or "document.pdf").name
    if Path(filename).suffix.lower() != ".pdf":
        await message.answer("Iltimos, faqat PDF fayl yuboring. 📄")
        return
    max_bytes = MAX_PDF_SIZE_MB * 1024 * 1024
    if doc.file_size and doc.file_size > max_bytes:
        await message.answer(f"PDF hajmi juda katta. Maksimal hajm: {MAX_PDF_SIZE_MB} MB.")
        return

    await message.answer("⏳ PDF'ni o'qiyapman...")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="studymate_", suffix=".pdf", delete=False) as temp:
            temp_path = temp.name
        file = await message.bot.get_file(doc.file_id)
        await message.bot.download_file(file.file_path, temp_path)

        text = await asyncio.to_thread(extract_text_from_pdf, temp_path)
        chunks = split_into_chunks(text)
        inserted = await asyncio.to_thread(add_document, message.from_user.id, chunks, filename)
        if not inserted:
            await message.answer("ℹ️ Bu PDF allaqachon yuklangan yoki undan yangi ma'lumot topilmadi.")
            return
        await message.answer(f"✅ PDF tayyor! {inserted} ta parcha indekslandi. Endi savol berishingiz mumkin.")
    except (OSError, ValueError, RuntimeError):
        logger.exception("PDF processing failed")
        await message.answer("❌ PDF'ni qayta ishlashda xatolik yuz berdi. Faylni tekshirib, qayta urinib ko'ring.")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.message(Command("test"))
async def cmd_test(message: types.Message):
    await message.answer("⏳ Test tayyorlanmoqda...")
    try:
        result = await asyncio.to_thread(generate_quiz, message.from_user.id)
        await message.answer(result)
    except (RuntimeError, ValueError, OSError):
        logger.exception("Quiz generation failed")
        await message.answer("❌ Test yaratishda xatolik yuz berdi. Keyinroq qayta urinib ko'ring.")


@router.message(Command("xulosa"))
async def cmd_summary(message: types.Message):
    await message.answer("⏳ Xulosa tayyorlanmoqda...")
    try:
        result = await asyncio.to_thread(generate_summary, message.from_user.id)
        await message.answer(result)
    except (RuntimeError, ValueError, OSError):
        logger.exception("Summary generation failed")
        await message.answer("❌ Xulosa yaratishda xatolik yuz berdi. Keyinroq qayta urinib ko'ring.")


@router.message(F.text)
async def handle_question(message: types.Message, state: FSMContext):
    if await state.get_state() is not None:
        return
    query = message.text.strip()
    if not query:
        return
    await message.answer("🤔 O'ylayapman...")
    try:
        chunks = await asyncio.to_thread(search_relevant_chunks, message.from_user.id, query)
        answer = await asyncio.to_thread(generate_answer, query, chunks)
        await message.answer(answer)
    except (RuntimeError, ValueError, OSError):
        logger.exception("Answer generation failed")
        await message.answer("❌ Javob tayyorlashda xatolik yuz berdi. Keyinroq qayta urinib ko'ring.")
