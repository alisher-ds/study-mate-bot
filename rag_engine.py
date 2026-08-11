import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from config import GROQ_API_KEY, AI_MODEL

embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
groq_client = Groq(api_key=GROQ_API_KEY)


def get_or_create_collection(user_id: int):
    collection_name = f"user_{user_id}"
    return chroma_client.get_or_create_collection(name=collection_name)


def add_document(user_id: int, chunks: list, doc_name: str):
    collection = get_or_create_collection(user_id)
    if not chunks:
        return
    ids, embeddings, metadatas, documents = [], [], [], []
    for index, chunk in enumerate(chunks):
        ids.append(f"{doc_name}_{index}")
        embeddings.append(embedding_model.encode(chunk).tolist())
        metadatas.append({"source": doc_name, "chunk_index": index})
        documents.append(chunk)
    collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)


def search_relevant_chunks(user_id: int, query: str, top_k: int = 3) -> list:
    collection = get_or_create_collection(user_id)
    if collection.count() == 0:
        return []
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    chunks = []
    for i, doc in enumerate(results['documents'][0]):
        chunks.append({"text": doc, "source": results['metadatas'][0][i].get("source", "noma'lum")})
    return chunks


def generate_answer(query: str, relevant_chunks: list) -> str:
    if relevant_chunks:
        context_text = "\n".join([f"[{c['source']}]: {c['text']}" for c in relevant_chunks])
        system_prompt = ("Sen StudyMate — do'stona o'quv yordamchisan. "
                        "Faqat berilgan kontekst (PDF matni) asosida javob ber. "
                        "O'zbek tilida, tushunarli va iliq javob ber.")
        user_prompt = f"Kontekst:\n{context_text}\n\nSavol: {query}\n\nYuqoridagi kontekst asosida javob ber."
    else:
        system_prompt = ("Sen StudyMate — do'stona, aqlli o'quv yordamchisan. "
                        "Foydalanuvchi hali PDF yuklamagan, shuning uchun o'z umumiy "
                        "biliming asosida javob ber. O'zbek tilida, samimiy va "
                        "foydali javob ber. Oxirida 'Aniqroq javob uchun PDF yuklashingiz mumkin' deb qo'sh.")
        user_prompt = f"Savol: {query}\n\nIltimos, javob ber."

    response = groq_client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content


def generate_quiz(user_id: int, num_questions: int = 5) -> str:
    collection = get_or_create_collection(user_id)
    if collection.count() == 0:
        return "Hozircha test tuzish uchun PDF yuklanmagan. Iltimos, avval PDF yuboring."
    all_docs = collection.get(limit=min(10, collection.count()))
    context = "\n".join(all_docs['documents'])
    response = groq_client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {"role": "system", "content": "Sen test tuzuvchisan. Berilgan matn asosida savollar tuz."},
            {"role": "user", "content": f"Quyidagi matn asosida {num_questions} ta test savoli tuz (javoblari bilan). O'zbek tilida.\n\n{context}"}
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content


def generate_summary(user_id: int) -> str:
    collection = get_or_create_collection(user_id)
    if collection.count() == 0:
        return "Hozircha xulosa chiqarish uchun PDF yuklanmagan."
    all_docs = collection.get(limit=min(15, collection.count()))
    context = "\n".join(all_docs['documents'])
    response = groq_client.chat.completions.create(
        model=AI_MODEL,
        messages=[
            {"role": "system", "content": "Sen matn xulosachisisan. Qisqa va aniq xulosa ber."},
            {"role": "user", "content": f"Quyidagi matnning qisqacha xulosasini o'zbek tilida yoz:\n\n{context}"}
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content
