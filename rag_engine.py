# RAG (Retrieval Augmented Generation) tizimi uchun asosiy modul
# Bu fayl vektorli qidiruv va ma'lumotlarni saqlash funksiyalarini bajaradi

import chromadb
from sentence_transformers import SentenceTransformer
import os

# Global o'zgaruvchilar - bir marta yuklanadi va butun dastur davomida ishlatiladi

# Embedding modeli - matnni vektorlarga aylantirish uchun
# 'paraphrase-multilingual-MiniLM-L12-v2' ko'p tillarni, jumladan o'zbekchani ham qo'llab-quvvatlaydi
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ChromaDB client - vektorli ma'lumotlar bazasiga ulanish uchun
# "./chroma_db" papkasida ma'lumotlar doimiy saqlanadi (persistent)
chroma_client = chromadb.PersistentClient(path="./chroma_db")


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
