"""
PDF fayllar bilan ishlash uchun modul.
Bu modul PDF fayllardan matn o'qish va uni kichik bo'laklarga (chunk'larga) ajratish funksiyalarini o'z ichiga oladi.
"""

from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    """
    PDF fayldan matnni chiqarib oladi.
    
    Args:
        file_path (str): PDF faylning yo'li
        
    Returns:
        str: Barcha sahifalardan olingan matn, har bir sahifa oldida "[SAHIFA N]" belgisi bilan
    """
    # PdfReader obyekti orqali PDF faylni ochamiz
    reader = PdfReader(file_path)
    
    # Natijaviy matnni saqlash uchun bo'sh ro'yxat
    pages_text = []
    
    # Har bir sahifani aylanib chiqamiz
    # enumerate() funksiyasi indeks (sahifa raqami) va qiymat (sahifa obyekti) beradi
    for page_num, page in enumerate(reader.pages, start=1):
        # Sahifadan matnni chiqaramiz
        text = page.extract_text()
        
        # Har bir sahifa oldiga "[SAHIFA N]" belgisini qo'shamiz
        # Bu keyinchalik javob berganda manbani ko'rsatish uchun kerak
        pages_text.append(f"[SAHIFA {page_num}]\n{text}")
    
    # Barcha sahifalarni bitta string qilib birlashtiramiz
    # "\n\n" bilan ajratamiz, shunda sahifalar aniq ajralib turadi
    return "\n\n".join(pages_text)


def split_into_chunks(text: str, max_chunk_size: int = 300, min_chunk_size: int = 50) -> list:
    """
    Matnni paragraflar chegarasidan hurmat qilib bo'laklarga bo'ladi.
    So'zlar o'rtasida emas, balki paragraf (bo'sh qator) chegarasida kesadi,
    shunda ta'rif yoki fikr bo'linib qolmaydi.
    """
    # Matnni paragraflarga ajratamiz (bo'sh qator orqali)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # Agar joriy chunk + yangi paragraf max_chunk_size dan oshsa
        word_count = len((current_chunk + " " + para).split())
        
        if word_count > max_chunk_size and len(current_chunk.split()) >= min_chunk_size:
            # Joriy chunkni saqlaymiz, yangisini boshlaymiz
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            # Joriy chunkga qo'shamiz
            current_chunk = (current_chunk + "\n\n" + para).strip() if current_chunk else para
        
        # Agar bitta paragrafning o'zi juda uzun bo'lsa (masalan max_chunk_size dan 2 barobar katta),
        # uni alohida so'zlar bo'yicha bo'lamiz
        if len(para.split()) > max_chunk_size * 2:
            words = para.split()
            for i in range(0, len(words), max_chunk_size):
                sub_chunk = " ".join(words[i:i + max_chunk_size])
                chunks.append(sub_chunk)
            current_chunk = ""
    
    # Oxirgi qolgan chunkni ham qo'shamiz
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks
