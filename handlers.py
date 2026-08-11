"""
handlers.py — Botning barcha handler'lari (buyruqlar va mantiq).
- /start → do'stona kutib olish + ro'yxatdan o'tish (ism, telefon, shahar)
- PDF yuklash → matnni parchalab, vektor bazaga saqlash
- Savol → PDF bo'lsa shu asosida, bo'lmasa umumiy AI javob
- /test, /xulosa → test va xulosa
"""
from aiogram import Router, F, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

from database import add_user, get_user
from pdf_processor import extract_text_from_pdf, split_into_chunks
from rag_engine import add_document, search_relevant_chunks, generate_answer, generate_quiz, generate_summary

router = Router()


# ===== FSM: Ro'yxatdan o'tish holatlari (ketma-ket savol berish) =====
class Registration(StatesGroup):
    ism = State()       # 1-qadam: ism
    telefon = State()   # 2-qadam: telefon
    shahar = State()    # 3-qadam: shahar


# Telefon raqamni jo'natish tugmasi
phone_keyboard = ReplyKeyboardMarkup(
    keyboard=[[KeyboardButton(text=" Telefonimni jo'natish", request_contact=True)]],
    resize_keyboard=True
)


# ===== /start — Do'stona kutib olish + ro'yxatdan o'tish =====
@router.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    user_id = message.from_user.id
    existing_user = get_user(user_id)

    # Agar allaqachon ro'yxatdan o'tgan bo'lsa — oddiy salom
    if existing_user:
        ism = existing_user[1]
        await message.answer(
            f"Assalomu alaykum, {ism}! 👋 Yana xush kelibsiz!\n\n"
            " PDF yuklashingiz, savol berishingiz, /test yoki /xulosa buyruqlarini ishlatishingiz mumkin."
        )
        return

    # Yangi foydalanuvchi — ro'yxatdan o'tishni boshlaymiz
    await message.answer(
        "Assalomu alaykum! 🌟 StudyMate botiga xush kelibsiz!\n\n"
        "Men sizning o'quv yordamchingizman — PDF asosida savollarga javob beraman, "
        "test tuzaman, xulosa chiqaraman. Istalgan savolingizga ham javob beraman! 😊\n\n"
        "Keling, avval tanishib olaylik. Ismingiz nima?"
    )
    await state.set_state(Registration.ism)  # 1-holatga o'tamiz


# ===== 1-qadam: Ismni qabul qilish =====
@router.message(Registration.ism)
async def process_ism(message: types.Message, state: FSMContext):
    await state.update_data(ism=message.text.strip())
    await message.answer(
        f"Yoqimli ism ekan, {message.text.strip()}! 😊\n"
        "Endi telefon raqamingizni jo'nating (pastdagi tugmani bosing):",
        reply_markup=phone_keyboard
    )
    await state.set_state(Registration.telefon)


# ===== 2-qadam: Telefonni qabul qilish =====
@router.message(Registration.telefon, F.contact)
async def process_telefon(message: types.Message, state: FSMContext):
    await state.update_data(telefon=message.contact.phone_number)
    await message.answer(
        "Rahmat! 📱 Qayerda yashaysiz? (shahringizni yozing):"
    )
    await state.set_state(Registration.shahar)


# ===== 3-qadam: Shaharni qabul qilish + bazaga saqlash =====
@router.message(Registration.shahar)
async def process_shahar(message: types.Message, state: FSMContext):
    data = await state.get_data()
    user_id = message.from_user.id

    # Ma'lumotlarni bazaga saqlaymiz
    add_user(
        user_id=user_id,
        ism=data['ism'],
        telefon=data['telefon'],
        shahar=message.text.strip()
    )

    await state.clear()  # FSM ni tozalaymiz
    await message.answer(
        f"Barakalloh, {data['ism']}! 🎉 Ro'yxatdan muvaffaqiyatli o'tdingiz!\n\n"
        f"📍 Shahar: {message.text.strip()}\n\n"
        "Endi siz tayyorsiz! 🚀\n"
        "• PDF fayl yuboring — men uni o'rganaman\n"
        "• Istalgan savol bering — javob beraman\n"
        "• /test — test savollari\n"
        "• /xulosa — matn xulosasi"
    )


# ===== PDF yuklash =====
@router.message(F.document)
async def handle_pdf(message: types.Message):
    doc = message.document

    if not doc.file_name.endswith('.pdf'):
        await message.answer("Iltimos, faqat PDF fayl yuboring. 📄")
        return

    await message.answer("⏳ PDF ingizni o'qiyapman, biroz kuting...")

    # Faylni yuklab olamiz
    file = await message.bot.get_file(doc.file_id)
    file_path = f"temp_{doc.file_name}"
    await message.bot.download_file(file.file_path, file_path)

    try:
        # Matnni chiqaramiz va parchalaymiz
        text = extract_text_from_pdf(file_path)
        chunks = split_into_chunks(text)

        # Vektor bazaga saqlaymiz
        add_document(user_id=message.from_user.id, chunks=chunks, doc_name=doc.file_name)

        await message.answer(
            f"✅ PDF muvaffaqiyatli yuklandi! ({len(chunks)} ta parcha)\n"
            f"📄 {doc.file_name}\n\n"
            "Endi bu fayl bo'yicha savol berishingiz mumkin! 🎯"
        )
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")
    finally:
        # Vaqtinchalik faylni o'chiramiz
        import os
        if os.path.exists(file_path):
            os.remove(file_path)


# ===== SAVOL-JAVOB (PDF bo'lsa — shu asosida, bo'lmasa — umumiy AI) =====
@router.message(F.text)
async def handle_question(message: types.Message, state: FSMContext):
    # Agar ro'yxatdan o'tish jarayonida bo'lsa — savolni qabul qilmaymiz
    current_state = await state.get_state()
    if current_state is not None:
        return

    query = message.text.strip()
    await message.answer("🤔 O'ylayapman...")

    try:
        # Avval PDF dan mos parchalarni qidiramiz
        relevant_chunks = search_relevant_chunks(user_id=message.from_user.id, query=query)

        # generate_answer o'zi hal qiladi: PDF bormi-yo'qmi
        answer = generate_answer(query=query, relevant_chunks=relevant_chunks)

        await message.answer(answer)
    except Exception as e:
        await message.answer(f"❌ Xatolik yuz berdi: {e}")


# ===== /test — Test savollari =====
@router.message(Command("test"))
async def cmd_test(message: types.Message):
    await message.answer("⏳ Test tayyorlayapman...")
    try:
        quiz = generate_quiz(user_id=message.from_user.id)
        await message.answer(quiz)
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")


# ===== /xulosa — Matn xulosasi =====
@router.message(Command("xulosa"))
async def cmd_summary(message: types.Message):
    await message.answer("⏳ Xulosa chiqaryapman...")
    try:
        summary = generate_summary(user_id=message.from_user.id)
        await message.answer(summary)
    except Exception as e:
        await message.answer(f"❌ Xatolik: {e}")
