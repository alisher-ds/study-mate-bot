import asyncio
import logging
import os
import tempfile

from aiogram import F, Router, types
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove
from pypdf.errors import PdfReadError

from database import add_user, get_user
from pdf_processor import extract_text_from_pdf, split_into_chunks
from rag_engine import (
    add_document,
    generate_answer,
    generate_quiz,
    generate_summary,
    reset_user_data,
    search_relevant_chunks,
)

logger = logging.getLogger(__name__)

# NOTE: aiogram tries handlers in the order they are registered. The catch-all text
# handler (handle_question) must stay LAST, otherwise it swallows every command.
router = Router()

MAX_PDF_BYTES = 20 * 1024 * 1024  # Telegram Bot API does not let bots download larger files

GENERIC_ERROR = "😔 Kechirasiz, xatolik yuz berdi. Iltimos, birozdan keyin qayta urinib ko'ring."

HELP_TEXT = (
    "📚 StudyMate — o'quv yordamchisi\n\n"
    "• PDF fayl yuboring — men uni o'rganaman\n"
    "• Istalgan savol bering — PDF bo'lsa shu asosida, bo'lmasa umumiy bilim asosida "
    "javob beraman\n"
    "• /test — PDF asosida test savollari\n"
    "• /xulosa — PDF xulosasi\n"
    "• /reset — yuklangan barcha PDF ma'lumotlarimni o'chirish\n"
    "• /help — shu yordam xabari"
)


class Registration(StatesGroup):
    ism = State()
    telefon = State()
    shahar = State()


phone_keyboard = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text="📱 Telefonimni jo'natish", request_contact=True)]],
    resize_keyboard=True
)


# ----------------------------------------------------------------------------- commands

@router.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    existing_user = get_user(user_id)

    if existing_user:
        ism = existing_user[1]
        await message.answer(
            f"Assalomu alaykum, {ism}! 👋 Yana xush kelibsiz!\n\n"
            "📄 PDF yuklashingiz, savol berishingiz, /test yoki /xulosa buyruqlarini "
            "ishlatishingiz mumkin. Barcha buyruqlar: /help"
        )
        return

    await message.answer(
        "Assalomu alaykum! 🌟 StudyMate botiga xush kelibsiz!\n\n"
        "Men sizning o'quv yordamchingizman — PDF asosida savollarga javob beraman, "
        "test tuzaman, xulosa chiqaraman. Istalgan savolingizga ham javob beraman! 😊\n\n"
        "Keling, avval tanishib olaylik. Ismingiz nima?"
    )
    await state.set_state(Registration.ism)


@router.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer(HELP_TEXT)


@router.message(Command("reset"))
async def cmd_reset(message: types.Message):
    try:
        deleted = await reset_user_data(message.from_user.id)
    except Exception:
        logger.exception("Reset failed for user %s", message.from_user.id)
        await message.answer(GENERIC_ERROR)
        return

    if deleted:
        await message.answer("🗑 Yuklangan barcha PDF ma'lumotlaringiz o'chirildi.")
    else:
        await message.answer("Sizda o'chiriladigan PDF ma'lumotlari yo'q.")


@router.message(Command("test"))
async def cmd_test(message: types.Message):
    await message.answer("⏳ Test tayyorlayapman...")
    try:
        quiz = await generate_quiz(user_id=message.from_user.id)
        await message.answer(quiz)
    except Exception:
        logger.exception("Quiz generation failed for user %s", message.from_user.id)
        await message.answer(GENERIC_ERROR)


@router.message(Command("xulosa"))
async def cmd_summary(message: types.Message):
    await message.answer("⏳ Xulosa chiqaryapman...")
    try:
        summary = await generate_summary(user_id=message.from_user.id)
        await message.answer(summary)
    except Exception:
        logger.exception("Summary generation failed for user %s", message.from_user.id)
        await message.answer(GENERIC_ERROR)


# ----------------------------------------------------------------------------- PDF upload

@router.message(F.document)
async def handle_pdf(message: types.Message):
    doc = message.document
    file_name = doc.file_name or ""  # Telegram may omit the file name
    is_pdf = file_name.lower().endswith(".pdf") or doc.mime_type == "application/pdf"
    if not is_pdf:
        await message.answer("Iltimos, faqat PDF fayl yuboring. 📄")
        return

    if doc.file_size and doc.file_size > MAX_PDF_BYTES:
        await message.answer("Fayl juda katta. Iltimos, 20 MB gacha bo'lgan PDF yuboring. 📄")
        return

    # Only a label for the user and the vector store, never part of a file path.
    doc_name = (file_name or "document.pdf")[:100]
    await message.answer("⏳ PDF ingizni o'qiyapman, biroz kuting...")

    try:
        file = await message.bot.get_file(doc.file_id)
        # The user-supplied name is never used in the path; the temp dir is removed afterwards.
        with tempfile.TemporaryDirectory(prefix="studymate_") as tmp_dir:
            pdf_path = os.path.join(tmp_dir, "upload.pdf")
            await message.bot.download_file(file.file_path, pdf_path)
            text = await asyncio.to_thread(extract_text_from_pdf, pdf_path)

        chunks = split_into_chunks(text)
        if not chunks:
            await message.answer(
                "⚠️ Bu PDF dan matn chiqmadi. Ehtimol, bu skanerlangan (rasm ko'rinishidagi) "
                "PDF. Hozircha faqat matnli PDF fayllar qo'llab-quvvatlanadi — matnni "
                "belgilab olish mumkin bo'lgan PDF yuboring."
            )
            return

        stored = await add_document(user_id=message.from_user.id, chunks=chunks, doc_name=doc_name)
        await message.answer(
            f"✅ PDF muvaffaqiyatli yuklandi! ({stored} ta parcha)\n"
            f"📄 {doc_name}\n\nEndi bu fayl bo'yicha savol berishingiz mumkin! 🎯"
        )
    except PdfReadError:
        logger.warning("Unreadable PDF from user %s", message.from_user.id, exc_info=True)
        await message.answer(
            "❌ PDF ni ochib bo'lmadi. U parol bilan himoyalangan yoki buzilgan bo'lishi mumkin."
        )
    except Exception:
        logger.exception("PDF processing failed for user %s", message.from_user.id)
        await message.answer(GENERIC_ERROR)


# ----------------------------------------------------------------------------- registration

@router.message(Registration.ism, F.text)
async def process_ism(message: types.Message, state: FSMContext):
    ism = message.text.strip()[:100]
    await state.update_data(ism=ism)
    await message.answer(
        f"Yoqimli ism ekan, {ism}! 😊\n"
        "Endi telefon raqamingizni jo'nating (pastdagi tugmani bosing):",
        reply_markup=phone_keyboard
    )
    await state.set_state(Registration.telefon)


@router.message(Registration.ism)
async def ism_expected(message: types.Message):
    await message.answer("Iltimos, ismingizni matn ko'rinishida yozing. 🙂")


@router.message(Registration.telefon, F.contact)
async def process_telefon(message: types.Message, state: FSMContext):
    if message.contact.user_id != message.from_user.id:
        await message.answer(
            "Iltimos, boshqa odamniki emas, o'zingizning raqamingizni jo'nating "
            "(pastdagi tugma orqali). 📱",
            reply_markup=phone_keyboard
        )
        return

    await state.update_data(telefon=message.contact.phone_number)
    await message.answer(
        "Rahmat! 📱 Qayerda yashaysiz? (shahringizni yozing):",
        reply_markup=ReplyKeyboardRemove()
    )
    await state.set_state(Registration.shahar)


@router.message(Registration.telefon)
async def telefon_expected(message: types.Message):
    await message.answer(
        "Telefon raqamni qo'lda yozish shart emas. Iltimos, pastdagi "
        "«📱 Telefonimni jo'natish» tugmasini bosing.",
        reply_markup=phone_keyboard
    )


@router.message(Registration.shahar, F.text)
async def process_shahar(message: types.Message, state: FSMContext):
    data = await state.get_data()
    user_id = message.from_user.id
    shahar = message.text.strip()[:100]
    add_user(user_id=user_id, ism=data['ism'], telefon=data['telefon'], shahar=shahar)
    await state.clear()
    await message.answer(
        f"Barakalloh, {data['ism']}! 🎉 Ro'yxatdan muvaffaqiyatli o'tdingiz!\n\n"
        f"📍 Shahar: {shahar}\n\n"
        "Endi siz tayyorsiz! 🚀\n"
        "• PDF fayl yuboring — men uni o'rganaman\n"
        "• Istalgan savol bering — javob beraman\n"
        "• /test — test savollari\n"
        "• /xulosa — matn xulosasi\n"
        "• /help — barcha buyruqlar"
    )


@router.message(Registration.shahar)
async def shahar_expected(message: types.Message):
    await message.answer("Iltimos, shahringiz nomini matn ko'rinishida yozing. 📍")


# ------------------------------------------------------------------- free text (keep LAST)

@router.message(F.text, StateFilter(None))
async def handle_question(message: types.Message):
    query = message.text.strip()
    await message.answer("🤔 O'ylayapman...")
    try:
        relevant_chunks = await search_relevant_chunks(user_id=message.from_user.id, query=query)
        answer = await generate_answer(query=query, relevant_chunks=relevant_chunks)
        await message.answer(answer)
    except Exception:
        logger.exception("Question handling failed for user %s", message.from_user.id)
        await message.answer(GENERIC_ERROR)
