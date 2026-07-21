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


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Matnni kichik bo'laklarga (chunk'larga) ajratadi.
    
    Bu funksiya RAG (Retrieval Augmented Generation) tizimi uchun muhim,
    chunki katta matnni birdaniga emas, balki kichik qismlarga bo'lib
    vektor bazasiga saqlaymiz va qidiramiz.
    
    Args:
        text (str): Bo'laklarga ajratiladigan matn
        chunk_size (int): Har bir bo'lakdagi so'zlar soni (default: 500)
        overlap (int): Qo'shni bo'laklar orasidagi umumiy so'zlar soni (default: 50)
                      Bu ma'no uzilib qolmasligi uchun kerak
        
    Returns:
        list: Matn bo'laklarining ro'yxati
    """
    # Matnni so'zlarga ajratamiz (bo'sh joy bo'yicha)
    words = text.split()
    
    # Agar matn juda qisqa bo'lsa, uni bitta chunk qilib qaytaramiz
    if len(words) <= chunk_size:
        return [text]
    
    # Bo'laklarni saqlash uchun bo'sh ro'yxat
    chunks = []
    
    # Matnni bo'laklarga ajratish tsikli
    # start_index - har bir yangi bo'lak qayerdan boshlanishini ko'rsatadi
    # Har bir iteratsiyada start_index (chunk_size - overlap) ga oshadi
    # Masalan: chunk_size=500, overlap=50 bo'lsa, har safar 450 ta so'zga siljiymiz
    for i in range(0, len(words), chunk_size - overlap):
        # Joriy bo'lak uchun so'zlarni olamiz
        # words[i : i + chunk_size] - i indeksdan boshlab chunk_size ta so'zni oladi
        chunk_words = words[i : i + chunk_size]
        
        # So'zlarni qayta stringga aylantiramiz
        chunk_text = " ".join(chunk_words)
        
        # Tayyor bo'lakni ro'yxatga qo'shamiz
        chunks.append(chunk_text)
        
        # Agar oxirgi so'zgacha yetib borsak, tsiklni to'xtatamiz
        if i + chunk_size >= len(words):
            break
    
    return chunks
