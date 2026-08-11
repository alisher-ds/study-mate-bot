"""Backward-compatible imports for document processing."""

from document_processor import extract_text as extract_text_from_pdf
from document_processor import split_into_chunks

__all__ = ["extract_text_from_pdf", "split_into_chunks"]
