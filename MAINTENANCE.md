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
- Temporary PDF files are cleared promptly after chunk ingestion.

- Text extraction ignores page headers and footers using vertical bounding box heuristics.
- Embedding dimensions correspond directly to the selected sentence-transformers model.
- Idle user document sessions expire automatically after 24 hours of inactivity.
- Chunk overlap of 15% is maintained to preserve boundary context across splits.
- Foreign key pragma is enabled on each SQLite connection initialization.
