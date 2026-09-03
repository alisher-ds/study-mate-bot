# StudyMate Bot

> **RAG-based AI Learning Assistant for Telegram.**

StudyMate is a Telegram bot that allows users to upload documents (PDFs) and ask contextual questions using Retrieval-Augmented Generation (RAG) powered by LLMs.

---

## Features

- **Document Ingestion**: Extracts and chunks text from uploaded PDF materials.
- **Vector Search / RAG**: Retrieves relevant context to answer user queries accurately.
- **Contextual Responses**: Powered by high-performance LLM backends (e.g. LLaMA 3.3).
- **User Management**: SQLite database tracking user sessions and activity.

---

## Tech Stack

- **Language**: Python 3.10+
- **Bot Framework**: Aiogram
- **LLM / Provider**: Groq API (`llama-3.3-70b-versatile`)
- **Database**: SQLite (`users.db`)

---

## Quickstart

1. **Clone repository and setup virtual environment**:
   ```bash
   git clone https://github.com/alisher-ds/study-mate-bot.git
   cd study-mate-bot
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Configure environment variables**:
   Create a `.env` file with:
   ```env
   BOT_TOKEN=your_telegram_bot_token
   GROQ_API_KEY=your_groq_api_key
   ```

3. **Run the bot**:
   ```bash
   python bot.py
   ```
---

## Troubleshooting

- If Groq API rate limit is reached, the bot gracefully requests users to retry after a short cooldown.

- **Multilingual Support**: Supports Uzbek, Russian, and English documents effortlessly.
