"""PDF text extraction and chunking utilities."""

from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from every page while preserving page markers."""
    reader = PdfReader(file_path)
    if reader.is_encrypted:
        raise ValueError("Bu PDF parol bilan himoyalangan va o'qib bo'lmaydi.")

    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[SAHIFA {page_num}]\n{text}")

    result = "\n\n".join(pages).strip()
    if not result:
        raise ValueError("PDF ichidan matn topilmadi. Bu skanerlangan yoki rasmli PDF bo'lishi mumkin.")
    return result


def split_into_chunks(text: str, max_chunk_size: int = 300, min_chunk_size: int = 50) -> list[str]:
    """Split text into bounded word-based chunks without dropping content."""
    if not text or max_chunk_size <= 0 or min_chunk_size < 0:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_words = 0

    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            continue

        if len(words) > max_chunk_size:
            flush()
            for start in range(0, len(words), max_chunk_size):
                piece = " ".join(words[start:start + max_chunk_size]).strip()
                if piece:
                    chunks.append(piece)
            continue

        if current_words + len(words) > max_chunk_size and current_words >= min_chunk_size:
            flush()

        current.append(paragraph)
        current_words += len(words)

    flush()
    return chunks
