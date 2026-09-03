# StudyMate Bot Maintenance Notes

## Components

- `bot.py`: Main polling loop and dispatcher setup.
- `handlers.py`: Message routers for commands, text, and PDF document uploads.
- `pdf_processor.py`: Text extraction and chunking pipeline.
- `rag_engine.py`: Vector search context retrieval and Groq LLM integration.
- `database.py`: SQLite interaction for user state and logs.
- Chroma vector embeddings are stored locally to minimize redundant reprocessing.
- PDF chunk size is calibrated to balance context density and latency.

- PyMuPDF handles page-by-page text extraction with encoding fallbacks.
- SQLite uses WAL mode to support seamless concurrent reads during document querying.
- Cosine distance thresholds filter out noisy chunks before LLM prompting.
