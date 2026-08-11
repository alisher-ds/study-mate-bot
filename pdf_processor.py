"""Backward-compatible PDF processing API."""

from document_processor import extract_text as extract_text_from_pdf
from document_processor import split_into_chunks as _split_into_chunks


def split_into_chunks(text: str, max_chunk_size: int = 300, min_chunk_size: int = 50) -> list[str]:
    """Preserve the legacy parameter names used by existing callers/tests."""
    return _split_into_chunks(text, max_words=max_chunk_size, min_words=min_chunk_size)


__all__ = ["extract_text_from_pdf", "split_into_chunks"]
