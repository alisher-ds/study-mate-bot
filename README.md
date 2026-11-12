# StudyMate Bot

A Telegram study assistant built on Retrieval-Augmented Generation (RAG). Send it a PDF, then ask
questions about it, generate a quiz, or get a summary. Without a PDF it answers from the LLM's
general knowledge. Bot messages are in Uzbek.

[![CI](https://github.com/alisher-ds/study-mate-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/alisher-ds/study-mate-bot/actions/workflows/ci.yml)

## What it does

- **Ask questions about your PDF** — the question is embedded, the 3 most similar chunks of your
  documents are retrieved from ChromaDB, and the LLM answers using only that context.
- **`/test`** — generates 5 quiz questions with answers from your uploaded material.
- **`/xulosa`** — writes a short summary of your uploaded material.
- **General questions** — if you have not uploaded anything, the bot answers from the model's own
  knowledge and says so.
- **Per-user storage** — every Telegram user gets their own ChromaDB collection (`user_<id>`).
  Re-uploading a file with the same name replaces the earlier copy.

## Architecture

```
Telegram ──► aiogram Dispatcher ──► handlers.py (commands, PDF upload, registration, free text)
                                        │
                  PDF upload            │            question / /test / /xulosa
                       ▼                │                       ▼
              pdf_processor.py          │                 rag_engine.py
        pypdf text extraction           │      embed query ─► ChromaDB search (top 3)
        paragraph-based chunking        │      or sample chunks evenly over the whole document
        (~300 words per chunk)          │                       │
                       └──────────► rag_engine.add_document     ▼
                          sentence-transformers (one batch)   Groq chat completion
                          ──► ChromaDB  ./chroma_db           (llama-3.3-70b-versatile,
                                                               retries on rate limit/timeout)
              database.py: SQLite users.db (registration profile)
```

- Embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers), so Uzbek, Russian and
  English text can be searched together.
- The blocking calls (embedding, ChromaDB, Groq, PDF parsing) run in worker threads
  (`asyncio.to_thread`), so the bot keeps responding while one user is being served.
- PDF text is placed in a delimited `<context>` block and the prompt tells the model not to follow
  instructions found inside it (a mitigation against prompt injection, not a guarantee).

## Commands

| Command   | Description                                                   |
|-----------|---------------------------------------------------------------|
| `/start`  | Greets returning users; new users go through registration     |
| `/help`   | Lists what the bot can do                                     |
| `/test`   | Quiz questions from your uploaded PDFs                        |
| `/xulosa` | Summary of your uploaded PDFs                                 |
| `/reset`  | Deletes your uploaded-document data (your ChromaDB collection) |

Any PDF you send is ingested; any other text is treated as a question.

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/alisher-ds/study-mate-bot.git
cd study-mate-bot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in the values
python bot.py
```

The first run downloads the embedding model from Hugging Face, so it needs internet access and
takes a while.

### Environment variables

| Name           | Purpose                                              |
|----------------|------------------------------------------------------|
| `BOT_TOKEN`    | Telegram bot token from [@BotFather](https://t.me/BotFather) |
| `GROQ_API_KEY` | API key from the [Groq console](https://console.groq.com)    |

`bot.py` exits with a clear message if either is missing. The model name (`AI_MODEL`) and database
file (`DB_NAME`) are constants in `config.py`. Runtime data is written next to where you start the
bot: `users.db` and `chroma_db/` (both git-ignored).

## Limitations

- **Scanned PDFs are not supported.** There is no OCR; a PDF without a text layer is rejected with
  an explanation.
- **Free Groq limits.** On the free tier you can hit rate limits. The bot retries a few times
  (3 attempts, 30 s timeout each) and then asks the user to try again later.
- **File size:** at most 20 MB (the Telegram Bot API limit for bot downloads).
- **Chunk size vs. embedding model:** chunks are about 300 words, but the embedding model reads
  only a limited number of tokens per text (128 by default), so retrieval mostly "sees" the start
  of each chunk. Smaller chunks would likely retrieve better; this has not been tuned or measured.
- **`/test` and `/xulosa` use a sample.** They send at most 10 (quiz) or 15 (summary) chunks,
  spread evenly across your documents — not the full text.
- **No conversation memory.** Each question is answered independently.
- **No source citations** are guaranteed in answers.
- Long polling only, single process, no rate limiting of users, no answer-quality evaluation.

## Privacy

During registration the bot stores your **name, phone number (via Telegram's "share contact"
button), city, and registration time** in a local SQLite file, `users.db`.

- It is collected as part of the user profile at sign-up. Nothing else in the code uses the phone
  number today, so if you do not need it, remove the phone step.
- The file is **not encrypted** and is not uploaded anywhere by the bot. Protect the machine that
  runs it; `users.db` is in `.gitignore`.
- Only your own contact card is accepted for the phone step.
- Your questions and the retrieved/sampled PDF text are sent to Groq to produce answers; the phone
  number is not.
- `/reset` deletes your document vectors from ChromaDB. It does **not** delete your profile in
  `users.db`.

## Development

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The tests need no API keys, no network and no model download: the embedding model, ChromaDB and Groq
are replaced by fakes. They cover chunking, chunk-id uniqueness, handler order (commands are
registered before the free-text handler), PDF upload handling, Groq retry/error handling and the
prompt structure. CI (`.github/workflows/ci.yml`) runs `ruff` and `pytest` with read-only
permissions.

Not covered by tests: real Telegram, real Groq responses and real ChromaDB/sentence-transformers
behaviour.

## License

[MIT](LICENSE)

---

## Qisqacha (O'zbekcha)

**StudyMate** — Telegram uchun o'quv yordamchi bot. PDF yuboring, so'ng u bo'yicha savol bering,
`/test` bilan test savollari yoki `/xulosa` bilan qisqacha xulosa oling. PDF yuklanmagan bo'lsa,
bot umumiy bilim asosida javob beradi.

- **Buyruqlar:** `/start`, `/help`, `/test`, `/xulosa`, `/reset` (yuklangan PDF ma'lumotlarini
  o'chiradi).
- **Ishga tushirish:** `pip install -r requirements.txt`, `.env` faylida `BOT_TOKEN` va
  `GROQ_API_KEY` ni to'ldiring, so'ng `python bot.py`.
- **Cheklovlar:** skanerlangan (rasmli) PDF qo'llab-quvvatlanmaydi; fayl 20 MB gacha; bepul Groq
  limitiga yetib qolish mumkin; bot oldingi savollarni eslab qolmaydi.
- **Maxfiylik:** ro'yxatdan o'tishda ism, telefon raqam va shahar `users.db` (SQLite) faylida
  shifrlanmagan holda saqlanadi. Telefon raqam hozircha boshqa joyda ishlatilmaydi. Savollaringiz va
  PDF matnining bir qismi javob olish uchun Groq'ga yuboriladi. `/reset` profilni emas, faqat PDF
  ma'lumotlarini o'chiradi.
- **Testlar:** `pip install -r requirements-dev.txt`, so'ng `ruff check .` va `pytest`. Testlar API
  kalit va internetsiz ishlaydi.
