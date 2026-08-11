import hashlib
from functools import lru_cache

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

from config import AI_MODEL, CHROMA_PATH, GROQ_API_KEY


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@lru_cache(maxsize=1)
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


@lru_cache(maxsize=1)
def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)


def get_or_create_collection(user_id: int):
    return get_chroma_client().get_or_create_collection(name=f"user_{user_id}")


def add_document(user_id: int, chunks: list, doc_name: str) -> int:
    if not chunks:
        return 0
    collection = get_or_create_collection(user_id)
    model = get_embedding_model()
    ids, embeddings, metadatas, documents = [], [], [], []
    for index, chunk in enumerate(chunks):
        digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:16]
        ids.append(f"{digest}_{index}")
        documents.append(chunk)
        metadatas.append({"source": doc_name, "chunk_index": index})
    existing = set(collection.get(ids=ids, include=[]).get("ids", []))
    new_positions = [i for i, item in enumerate(ids) if item not in existing]
    if not new_positions:
        return 0
    new_ids = [ids[i] for i in new_positions]
    new_docs = [documents[i] for i in new_positions]
    new_meta = [metadatas[i] for i in new_positions]
    embeddings = model.encode(new_docs, normalize_embeddings=True).tolist()
    collection.add(ids=new_ids, embeddings=embeddings, metadatas=new_meta, documents=new_docs)
    return len(new_ids)


def search_relevant_chunks(user_id: int, query: str, top_k: int = 4) -> list:
    query = query.strip()
    if not query:
        return []
    collection = get_or_create_collection(user_id)
    count = collection.count()
    if count == 0:
        return []
    n_results = min(max(top_k, 1), count)
    embedding = get_embedding_model().encode(query, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=n_results)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    return [{"text": doc, "source": (metadatas[i] or {}).get("source", "noma'lum")} for i, doc in enumerate(documents)]


def _chat(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    response = get_groq_client().chat.completions.create(
        model=AI_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return content.strip() if content else "Kechirasiz, javob yaratib bo'lmadi."


def generate_answer(query: str, relevant_chunks: list) -> str:
    if relevant_chunks:
        context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in relevant_chunks)
        system = "Sen StudyMate — o'quv yordamchisisan. Faqat berilgan PDF kontekstiga tayangan holda javob ber. Kontekstda javob bo'lmasa, buni ochiq ayt va fakt to'qima. O'zbek tilida aniq javob ber."
        prompt = f"Kontekst:\n{context}\n\nSavol: {query}"
    else:
        system = "Sen StudyMate — do'stona va foydali o'quv yordamchisisan. Umumiy bilim asosida O'zbek tilida tushunarli javob ber."
        prompt = query
    return _chat(system, prompt)


def _get_context(user_id: int, limit: int = 12) -> str:
    collection = get_or_create_collection(user_id)
    if collection.count() == 0:
        return ""
    docs = collection.get(limit=min(limit, collection.count()), include=["documents"])
    return "\n\n".join(docs.get("documents", []))


def generate_quiz(user_id: int, num_questions: int = 5) -> str:
    context = _get_context(user_id)
    if not context:
        return "Hozircha test tuzish uchun PDF yuklanmagan. Iltimos, avval PDF yuboring."
    return _chat("Sen test tuzuvchisan. Faqat berilgan matndan foydalan. O'zbek tilida test tuz. Har bir savolda 4 variant va to'g'ri javob bo'lsin.", f"Matn:\n{context}\n\n{num_questions} ta test tuz.", 0.4)


def generate_summary(user_id: int) -> str:
    context = _get_context(user_id, limit=20)
    if not context:
        return "Hozircha xulosa chiqarish uchun PDF yuklanmagan."
    return _chat("Sen akademik matn xulosachisisan. Berilgan matnni O'zbek tilida qisqa, aniq va mazmunli xulosa qil.", f"Matn:\n{context}", 0.3)
