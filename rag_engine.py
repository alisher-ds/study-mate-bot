from dotenv import load_dotenv
load_dotenv()

# RAG (Retrieval Augmented Generation) tizimi uchun asosiy modul
# Bu fayl vektorli qidiruv va ma'lumotlarni saqlash funksiyalarini bajaradi

import chromadb
from sentence_transformers import SentenceTransformer
import os
from groq import Groq

# Global o'zgaruvchilar - bir marta yuklanadi va butun dastur davomida ishlatiladi

# Embedding modeli - matnni vektorlarga aylantirish uchun
# 'paraphrase-multilingual-MiniLM-L12-v2' ko'p tillarni, jumladan o'zbekchani ham qo'llab-quvvatlaydi
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ChromaDB client - vektorli ma'lumotlar bazasiga ulanish uchun
# "./chroma_db" papkasida ma'lumotlar doimiy saqlanadi (persistent)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Groq client - LLM (Large Language Model) orqali javob generatsiya qilish uchun
# API kalit .env fayldan olinadi
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_or_create_collection(user_id: int):
    """
    Har bir foydalanuvchi uchun alohida ChromaDB collection yaratadi yoki mavjudini qaytaradi.
    Collection nomi "user_{user_id}" formatida bo'ladi, shu orqali har bir foydalanuvchining
    ma'lumotlari ajratilgan holda saqlanadi.

    :param user_id: Telegram foydalanuvchisining unikal ID raqami
    :return: ChromaDB collection obyekti
    """
    # Collection nomini shakllantiramiz
    collection_name = f"user_{user_id}"
    
    # Agar collection mavjud bo'lsa, uni olamiz, yo'q bo'lsa yangisini yaratamiz
    # get_or_create_collection metodi avtomatik ravishda tekshirib beradi
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    return collection


def add_document(user_id: int, chunks: list, doc_name: str):
    """
    Matn bo'laklarini (chunks) vektorlarga aylantirib, ChromaDB ga qo'shadi.
    Har bir chunk uchun unikal ID va metadata yaratiladi.

    :param user_id: Telegram foydalanuvchisining ID raqami
    :param chunks: Matndan ajratilgan bo'laklar ro'yxati (har biri string)
    :param doc_name: Hujjat nomi (manba sifatida ishlatiladi)
    """
    # Foydalanuvchi uchun collection ni olamiz yoki yaratamiz
    collection = get_or_create_collection(user_id)
    
    # Agar chunks bo'sh bo'lsa, hech narsa qilmaslik
    if not chunks:
        return
    
    # Har bir chunk uchun ID, embedding va metadata tayyorlaymiz
    ids = []
    embeddings = []
    metadatas = []
    
    for index, chunk in enumerate(chunks):
        # Har bir chunk uchun unikal ID yaratamiz: "{hujjat_nomi}_{indeks}"
        chunk_id = f"{doc_name}_{index}"
        ids.append(chunk_id)
        
        # Chunk matnini vektorga aylantiramiz (embedding)
        # embedding_model.encode() matnni vektorga o'giradi
        embedding = embedding_model.encode(chunk)
        embeddings.append(embedding.tolist())  # NumPy array ni list ga o'tkazamiz
        
        # Metadata - qo'shimcha ma'lumotlar (manba va indeks)
        metadata = {
            "source": doc_name,      # Qaysi fayldan olinganligi
            "chunk_index": index     # Chunk tartib raqami
        }
        metadatas.append(metadata)
    
    # Barcha ma'lumotlarni ChromaDB collection ga qo'shamiz
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=chunks  # Asl matn ham saqlanadi (ixtiyoriy, lekin foydali)
    )


def search_relevant_chunks(user_id: int, query: str, top_k: int = 3) -> list:
    """
    Foydalanuvchi savoliga eng mos keladigan chunk'larni qidiradi.
    Savolni vektorga aylantirib, ChromaDB dan eng yaqin vektorlarni topadi.

    :param user_id: Telegram foydalanuvchisining ID raqami
    :param query: Foydalanuvchining savoli (string)
    :param top_k: Qancha ta eng mos natijani qaytarish kerakligi (default: 3)
    :return: [{"text": chunk_matni, "source": manba_nomi}, ...] formatidagi ro'yxat
    """
    # Foydalanuvchi collection ini olamiz
    collection = get_or_create_collection(user_id)
    
    # Savolni vektorga aylantiramiz (embedding)
    query_embedding = embedding_model.encode(query)
    
    # ChromaDB dan eng mos top_k ta chunk'ni qidiramiz
    # query_embeddings - qidiruv vektori
    # n_results - qancha natija qaytarish kerakligi
    # include=["documents", "metadatas"] - matn va metadata'larni qaytarish
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    
    # Natijalarni qulay formatga o'tkazamiz
    relevant_chunks = []
    
    # results['documents'][0] - topilgan chunk matnlari ro'yxati
    # results['metadatas'][0] - har bir chunk uchun metadata ro'yxati
    documents = results['documents'][0] if results['documents'] else []
    metadatas = results['metadatas'][0] if results['metadatas'] else []
    
    # Har bir natijani {"text": ..., "source": ...} formatiga keltiramiz
    for i, doc in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}
        source = metadata.get("source", "noma'lum")  # Agar source bo'lmasa, "noma'lum" deb yozamiz
        
        relevant_chunks.append({
            "text": doc,
            "source": source
        })
    
    return relevant_chunks


def generate_answer(query: str, relevant_chunks: list) -> str:
    """
    Topilgan kontekst (relevant_chunks) asosida foydalanuvchi savoliga javob generatsiya qiladi.
    Groq API orqali LLM (Llama-3.3-70b) modelidan foydalanadi.

    :param query: Foydalanuvchining savoli
    :param relevant_chunks: Qidiruv natijasida topilgan kontekst parchalari ro'yxati
    :return: Generatsiya qilingan javob (string)
    """
    # 1. Barcha chunk'larni birlashtiramiz, har biriga manba nomini qo'shib
    context_parts = []
    for chunk in relevant_chunks:
        text = chunk.get("text", "")
        source = chunk.get("source", "noma'lum")
        # Har bir chunk'ni "[fayl_nomi]: matn" formatida qo'shamiz
        context_parts.append(f"[{source}]: {text}")
    
    # Barcha kontekstlarni bitta matnga birlashtiramiz
    context_text = "\n\n".join(context_parts)
    
    # 2. Prompt tuzamiz - LLM ga aniq ko'rsatma beramiz
    system_prompt = """Siz yordamchi assistantsiz. Faqat berilgan kontekst (matn) asosida javob bering.
Agar javob kontekstda topilmasa, aniq "Bu ma'lumot faylda topilmadi" deb ayting.
O'zbek tilida javob bering."""

    user_prompt = f"""Kontekst:
{context_text}

Foydalanuvchi savoli: {query}

Yuqoridagi kontekst asosida savolga javob bering."""

    # 3. Groq API ga so'rov yuboramiz
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Ishlatiladigan LLM modeli
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,  # Past qiymat - aniq va faktik javoblar uchun
    )
    
    # 4. Javobni qaytaramiz
    answer = response.choices[0].message.content
    return answer


def generate_quiz(user_id: int, num_questions: int = 5) -> str:
    """
    Foydalanuvchining yuklagan hujjatlari asosida test savollari generatsiya qiladi.
    
    Args:
        user_id (int): Foydalanuvchi ID raqami
        num_questions (int): Generatsiya qilinadigan savollar soni (default: 5)
    
    Returns:
        str: Test savollari va javob variantlari matni
    """
    # Foydalanuvchining collection'ini olamiz
    collection = get_or_create_collection(user_id)
    
    # Collection'dan barcha saqlangan chunk'larni olamiz (limit bilan)
    # Katta hajmli ma'lumotlar uchun birinchi 10 ta chunk'ni namuna sifatida olamiz
    results = collection.get(limit=10, include=["documents"])
    
    documents = results['documents']
    
    # Agar hech qanday ma'lumot bo'lmasa, xabar qaytaramiz
    if not documents:
        return "Hozircha test tuzish uchun yetarli ma'lumot yo'q."
    
    # Birinchi 10 ta chunk'ni birlashtiramiz
    context_text = "\n\n".join(documents)
    
    # Groq uchun prompt tuzamiz
    system_prompt = (
        "Siz o'qituvchi yordamchisisiz. Sizga berilgan matn asosida test savollari tuzishingiz kerak. "
        "Har bir savol 4 ta variantdan (a, b, c, d) iborat bo'lsin va to'g'ri javobni aniq ko'rsating. "
        "Javoblarni o'zbek tilida yozing."
    )
    
    user_prompt = (
        f"Quyidagi matn asosida {num_questions} ta test savoli tuz:\n\n"
        f"{context_text}"
    )
    
    # Groq API ga so'rov yuboramiz
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.5,  # Biroz kreativlik uchun o'rtacha temperatura
    )
    
    return chat_completion.choices[0].message.content


def generate_summary(user_id: int) -> str:
    """
    Foydalanuvchining yuklagan hujjatlari asosida qisqa xulosa (summary) tayyorlaydi.
    
    Args:
        user_id (int): Foydalanuvchi ID raqami
    
    Returns:
        str: Hujjatning 5-7 ta asosiy fikrdan iborat qisqacha xulosasi
    """
    # Foydalanuvchining collection'ini olamiz
    collection = get_or_create_collection(user_id)
    
    # Collection'dan barcha mavjud chunk'larni olamiz
    # Eslatma: limit=None deb barchasini olsak ham, xotira cheklovlari bo'lishi mumkin
    results = collection.get(include=["documents"])
    
    documents = results['documents']
    
    # Agar hech qanday ma'lumot bo'lmasa, xabar qaytaramiz
    if not documents:
        return "Hozircha xulosa chiqarish uchun yetarli ma'lumot yo'q."
    
    # Barcha chunk'larni bitta matnga birlashtiramiz
    full_text = "\n\n".join(documents)
    
    # Agar matn juda uzun bo'lsa (masalan, 4000 belgidan oshsa), kesib olamiz
    # Bu token limitlaridan oshib ketmaslik uchun kerak
    if len(full_text) > 4000:
        full_text = full_text[:4000] + "..."
    
    # Groq uchun prompt tuzamiz
    system_prompt = (
        "Siz matn tahlilchisisiz. Sizga berilgan matnni o'qib, uning asosiy g'oyalarini ajratib oling. "
        "Natijani 5-7 ta asosiy fikr ko'rinishida, o'zbek tilida qisqacha xulosa qilib bering."
    )
    
    user_prompt = f"Quyidagi matnni qisqacha xulosala:\n\n{full_text}"
    
    # Groq API ga so'rov yuboramiz
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.3,  # Aniq va faktik xulosa uchun past temperatura
    )
    
    return chat_completion.choices[0].message.content
