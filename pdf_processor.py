"""
PDF fayllar bilan ishlash uchun modul.
Bu modul PDF fayllardan matn o'qish va uni kichik bo'laklarga (chunk'larga) ajratadi.
"""

from pypdf import PdfReader


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


def split_into_chunks(text: str, max_chunk_size: int = 300, min_chunk_size: int = 50) -> list:
    """
    Matnni paragraflar chegarasidan hurmat qilib bo'laklarga bo'ladi.
    So'zlar o'rtasida emas, balki paragraf (bo'sh qator) chegarasida kesadi,
    shunda ta'rif yoki fikr bo'linib qolmaydi.

    max_chunk_size dan 2 barobar uzun paragraf so'zlar bo'yicha bo'linadi; bunda
    undan oldin yig'ilgan chunk avval alohida saqlanadi (yo'qolmaydi).
    Bo'sh matn uchun bo'sh ro'yxat qaytadi.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para_words = para.split()

        if len(para_words) > max_chunk_size * 2:
            # Avval yig'ilgan chunkni saqlab qo'yamiz, so'ng uzun paragrafni bo'lamiz
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            for i in range(0, len(para_words), max_chunk_size):
                chunks.append(" ".join(para_words[i:i + max_chunk_size]))
            continue

        current_words = len(current_chunk.split())
        overflow = current_words + len(para_words) > max_chunk_size
        if current_chunk and overflow and current_words >= min_chunk_size:
            # Joriy chunkni saqlaymiz, yangisini boshlaymiz
            chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
