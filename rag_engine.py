from dotenv import load_dotenv
load_dotenv()

# RAG (Retrieval Augmented Generation) tizimi uchun asosiy modul
# Bu fayl vektorli qidiruv va ma'lumotlarni saqlash funksiyalarini bajaradi

import chromadb
from sentence_transformers import SentenceTransformer
import os
from groq import Groq

embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_or_create_collection(user_id: int):
    collection_name = f"user_{user_id}"
    return chroma_client.get_or_create_collection(name=collection_name)


def add_document(user_id: int, chunks: list, doc_name: str):
    collection = get_or_create_collection(user_id)
    if not chunks:
        return
    ids, embeddings, metadatas = [], [], []
    for index, chunk in enumerate(chunks):
        ids.append(f"{doc_name}_{index}")
        embeddings.append(embedding_model.encode(chunk).tolist())
        metadatas.append({"source": doc_name, "chunk_index": index})
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=chunks)


def search_relevant_chunks(user_id: int, query: str, top_k: int = 3) -> list:
    collection = get_or_create_collection(user_id)
    query_embedding = embedding_model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    relevant_chunks = []
    documents = results['documents'][0] if results['documents'] else []
    metadatas = results['metadatas'][0] if results['metadatas'] else []
    for i, doc in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}
        relevant_chunks.append({"text": doc, "source": metadata.get("source", "noma'lum")})
    return relevant_chunks


def generate_answer(query: str, relevant_chunks: list) -> str:
    context_parts = []
    for chunk in relevant_chunks:
        text = chunk.get("text", "")
        source = chunk.get("source", "noma'lum")
        context_parts.append(f"[{source}]: {text}")
    context_text = "\n\n".join(context_parts)

    system_prompt = """Siz yordamchi assistantsiz. Faqat berilgan kontekst (matn) asosida javob bering.
Agar javob kontekstda topilmasa, aniq "Bu ma'lumot faylda topilmadi" deb ayting.
O'zbek tilida javob bering."""

    user_prompt = f"""Kontekst:
{context_text}

Foydalanuvchi savoli: {query}

Yuqoridagi kontekst asosida savolga javob bering."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def generate_quiz(user_id: int, num_questions: int = 5) -> str:
    collection = get_or_create_collection(user_id)
    all_data = collection.get()
    documents = all_data.get('documents', [])
    sample_text = " ".join(documents[:10])

    prompt = f"""Quyidagi matn asosida {num_questions} ta test savoli tuz, har biri 4 variantli (a,b,c,d), to'g'ri javobni ko'rsatib. O'zbek tilida yoz.

MATN:
{sample_text[:3000]}
"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content


def generate_summary(user_id: int) -> str:
    collection = get_or_create_collection(user_id)
    all_data = collection.get()
    documents = all_data.get('documents', [])
    full_text = " ".join(documents)

    prompt = f"""Quyidagi matnni 5-7 ta asosiy fikr bilan qisqacha xulosala, o'zbek tilida:

{full_text[:4000]}
"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
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
