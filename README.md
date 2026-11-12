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
        token-based chunking            │      or sample chunks evenly over the whole document
        (≤118 tokens, ~12% overlap)     │                       │
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

**Use Python 3.12.** It is the version this project is tested on (locally on Windows and in CI).
**Python 3.14 is not supported yet:** the pinned `aiogram==3.13.1` needs `aiohttp<3.11`, and that
release line has no wheel for 3.14 (pip would try to compile it). Other versions are untested.

No C++ compiler is needed: every dependency, including ChromaDB (`chromadb` 1.x, which replaced
the `chroma-hnswlib` package that had to be compiled), installs from prebuilt wheels. The install
downloads a lot of data because `sentence-transformers` pulls in PyTorch.

**Windows (PowerShell)**

```powershell
git clone https://github.com/alisher-ds/study-mate-bot.git
cd study-mate-bot
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env          # then fill in the values
.venv\Scripts\python.exe bot.py
```

**Linux / macOS**

```bash
git clone https://github.com/alisher-ds/study-mate-bot.git
cd study-mate-bot
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env            # then fill in the values
.venv/bin/python bot.py
```

If pip asks for "Microsoft Visual C++ 14.0 or greater", you are not on Python 3.12 or you are
installing an old `chromadb`; check with `.venv\Scripts\python.exe --version`.

The first run downloads the embedding model from Hugging Face, so it needs internet access and
takes a while.

### Upgrading from an older version

Earlier versions of this project used `chromadb` 0.5.x. Its `chroma_db/` folder is not migrated by
this project, so after upgrading **delete the `chroma_db/` folder and upload your PDFs again**
(`users.db` is separate and can stay). The same applies to documents uploaded before the
token-based chunking was introduced: they keep their old, longer chunks until re-uploaded.

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
- **Chunk size follows the embedding model's token limit.** The embedding model
  (`paraphrase-multilingual-MiniLM-L12-v2`) reads at most 128 tokens per text (`max_seq_length`,
  read from the loaded model) and silently drops the rest. Chunks are therefore measured in tokens
  with the model's own tokenizer, not in words: each chunk holds at most **118 tokens** (the limit
  minus 10 reserved for the 2 special tokens and a safety margin), and consecutive chunks repeat
  up to **14 tokens (about 12%)** from the end of the previous chunk. Paragraph boundaries are kept
  where they fit; a longer paragraph is split at sentence boundaries, then by words. Uzbek text
  costs about 2.1 tokens per word (English about 1.4, measured on a small sample), so a chunk is
  roughly 45–55 Uzbek or 65–80 English words. Retrieval quality with these settings has not been
  evaluated. Documents uploaded before this change keep their old, longer chunks until you upload
  them again.
- **`/test` and `/xulosa` use a sample.** They send at most 10 (quiz) or 15 (summary) chunks,
  spread evenly across your documents — not the full text. Because chunks are now short, this
  covers much less text than before.
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

```powershell
# Windows (use .venv/bin/python on Linux/macOS)
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m pytest
```

The tests need no API keys, no network and no embedding-model download. The embedding model and
Groq are replaced by fakes (chunking is tested with a stub tokenizer). ChromaDB is **real** in
`tests/test_chroma_integration.py`: a temporary directory, hand-made vectors and telemetry turned
off, covering add/query/get/delete, replacing a re-uploaded file and the `/reset` logic; the other
tests use an in-memory fake of it. Together they cover chunking (token limit, overlap, paragraph
and sentence boundaries), chunk-id uniqueness, handler order (commands are registered before the
free-text handler), PDF upload handling, Groq retry/error handling and the prompt structure. CI
(`.github/workflows/ci.yml`, Python 3.12) runs `ruff` and `pytest` with read-only permissions.

Not covered by tests: real Telegram, real Groq responses and real sentence-transformers behaviour.

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
