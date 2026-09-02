# StudyMate Bot Maintenance Notes

## Components

- `bot.py`: Main polling loop and dispatcher setup.
- `handlers.py`: Message routers for commands, text, and PDF document uploads.
- `pdf_processor.py`: Text extraction and chunking pipeline.
- `rag_engine.py`: Vector search context retrieval and Groq LLM integration.
- `database.py`: SQLite interaction for user state and logs.
