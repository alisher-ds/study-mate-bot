"""
PDF fayllar bilan ishlash uchun modul.
Bu modul PDF fayllardan matn o'qish va uni kichik bo'laklarga (chunk'larga) ajratadi.
"""

import re
from collections.abc import Callable

from pypdf import PdfReader

DEFAULT_OVERLAP_RATIO = 0.12  # chunklar orasida ~12% takror (10-15% oralig'ida)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def extract_text_from_pdf(file_path: str) -> str:
    """
    PDF fayldan matnni chiqarib oladi.

    Args:
        file_path (str): PDF faylning yo'li

    Returns:
        str: Matnli sahifalardan olingan matn, har bir sahifa oldida "[SAHIFA N]" belgisi bilan.
             Hech qaysi sahifada matn bo'lmasa (masalan, skanerlangan PDF) bo'sh satr qaytadi.
    """
    reader = PdfReader(file_path)

    pages_text = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        # Skanerlangan (rasmli) sahifalarda matn qatlami yo'q: bunday sahifalarni
        # o'tkazib yuboramiz, aks holda faqat "[SAHIFA N]" belgilari qolib ketadi.
        if not text:
            continue

        # Har bir sahifa oldiga "[SAHIFA N]" belgisini qo'shamiz
        pages_text.append(f"[SAHIFA {page_num}]\n{text}")

    # "\n\n" bilan ajratamiz, shunda sahifalar aniq ajralib turadi
    return "\n\n".join(pages_text)


def split_into_chunks(
    text: str,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> list:
    """
    Matnni embedding modeli sig'diradigan, token bo'yicha o'lchangan bo'laklarga bo'ladi.

    Har bir chunk `max_tokens` dan oshmaydi (model uzunroq matnni jimgina kesib tashlaydi).
    Chunk ikki qismdan iborat: oldingi chunkning oxiridan ko'chirilgan takror (overlap,
    `max_tokens * overlap_ratio` tokengacha) va yangi matn. Yangi matn paragraf chegarasida
    uziladi; sig'maydigan paragraf avval gaplarga, gap sig'masa so'zlarga bo'linadi.
    Bo'sh matn uchun bo'sh ro'yxat qaytadi.

    count_tokens: matndagi tokenlar sonini qaytaradi (maxsus tokenlarsiz).
    max_tokens: chunk uchun ruxsat etilgan eng ko'p token (maxsus tokenlar va zaxira ayirilgan).
    """
    if max_tokens < 2:
        raise ValueError("max_tokens must be at least 2")
    if not 0 <= overlap_ratio < 0.5:
        raise ValueError("overlap_ratio must be in [0, 0.5)")

    overlap_tokens = int(max_tokens * overlap_ratio)
    new_text_tokens = max_tokens - overlap_tokens  # takrordan keyin qolgan joy

    pieces = []
    for para in (p.strip() for p in text.split("\n\n")):
        if para:
            pieces.extend(_paragraph_pieces(para, new_text_tokens, count_tokens))

    chunks = []
    parts = []
    used = 0
    overlap = ""
    for piece, tokens in pieces:
        if parts and used + tokens > new_text_tokens:
            body = "\n\n".join(parts)
            _emit(chunks, f"{overlap} {body}" if overlap else body, max_tokens, count_tokens)
            overlap = _tail(body, overlap_tokens, count_tokens)
            parts, used = [], 0
        parts.append(piece)
        used += tokens

    if parts:
        body = "\n\n".join(parts)
        _emit(chunks, f"{overlap} {body}" if overlap else body, max_tokens, count_tokens)

    return chunks


def _paragraph_pieces(para: str, budget: int, count_tokens: Callable[[str], int]) -> list:
    """Paragrafni (matn, token soni) juftliklariga ajratadi; har biri budget ga sig'adi."""
    tokens = count_tokens(para)
    if tokens <= budget:
        return [(para, tokens)]

    pieces = []
    buffer = []
    buffer_tokens = 0

    def flush():
        nonlocal buffer, buffer_tokens
        if buffer:
            pieces.append((" ".join(buffer), buffer_tokens))
        buffer, buffer_tokens = [], 0

    for sentence in _SENTENCE_END.split(para):
        sentence_tokens = count_tokens(sentence)
        if sentence_tokens > budget:
            flush()
            pieces.extend(_word_pieces(sentence, budget, count_tokens))
            continue
        if buffer and buffer_tokens + sentence_tokens > budget:
            flush()
        buffer.append(sentence)
        buffer_tokens += sentence_tokens
    flush()
    return pieces


def _word_pieces(text: str, budget: int, count_tokens: Callable[[str], int]) -> list:
    """Bitta juda uzun gapni so'zlar bo'yicha bo'ladi (budget dan uzun so'z belgilar bo'yicha)."""
    pieces = []
    buffer = []
    buffer_tokens = 0

    def flush():
        nonlocal buffer, buffer_tokens
        if buffer:
            pieces.append((" ".join(buffer), buffer_tokens))
        buffer, buffer_tokens = [], 0

    for word in text.split():
        word_tokens = count_tokens(word)
        if word_tokens > budget:
            flush()
            for part in _hard_split(word, budget, count_tokens):
                pieces.append((part, count_tokens(part)))
            continue
        if buffer and buffer_tokens + word_tokens > budget:
            flush()
        buffer.append(word)
        buffer_tokens += word_tokens
    flush()
    return pieces


def _hard_split(word: str, budget: int, count_tokens: Callable[[str], int]) -> list:
    """Bo'sh joysiz uzun satrni (masalan, URL yoki base64) budget ga sig'guncha ikkiga bo'ladi."""
    if len(word) <= 1 or count_tokens(word) <= budget:
        return [word]
    mid = len(word) // 2
    return _hard_split(word[:mid], budget, count_tokens) + _hard_split(
        word[mid:], budget, count_tokens
    )


def _tail(text: str, budget: int, count_tokens: Callable[[str], int]) -> str:
    """Matn oxiridagi, budget tokenga sig'adigan so'zlar (keyingi chunk uchun takror)."""
    if budget <= 0:
        return ""
    tail = []
    used = 0
    for word in reversed(text.split()):
        word_tokens = count_tokens(word)
        if used + word_tokens > budget:
            break
        tail.append(word)
        used += word_tokens
    return " ".join(reversed(tail))


def _emit(chunks: list, text: str, max_tokens: int, count_tokens: Callable[[str], int]) -> None:
    """
    Chunkni ro'yxatga qo'shadi. Qismlarning token yig'indisi butun matnning haqiqiy token
    soniga teng bo'lmasligi mumkin (tokenizerga bog'liq), shuning uchun oxirgi tekshiruv:
    limitdan oshsa, matn yo'qolmasdan ikkiga bo'linadi.
    """
    if len(text) <= 1 or count_tokens(text) <= max_tokens:
        chunks.append(text)
        return
    words = text.split()
    if len(words) > 1:
        mid = len(words) // 2
        left, right = " ".join(words[:mid]), " ".join(words[mid:])
    else:
        mid = len(text) // 2
        left, right = text[:mid], text[mid:]
    _emit(chunks, left, max_tokens, count_tokens)
    _emit(chunks, right, max_tokens, count_tokens)
