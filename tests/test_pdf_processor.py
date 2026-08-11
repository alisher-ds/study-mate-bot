from pdf_processor import split_into_chunks


def test_split_empty_text():
    assert split_into_chunks("") == []


def test_split_preserves_short_paragraphs():
    text = "Birinchi paragraf.\n\nIkkinchi paragraf."
    chunks = split_into_chunks(text, max_chunk_size=20, min_chunk_size=1)
    assert len(chunks) == 1
    assert "Birinchi paragraf." in chunks[0]
    assert "Ikkinchi paragraf." in chunks[0]


def test_split_long_paragraph_is_bounded():
    text = " ".join(f"word{i}" for i in range(120))
    chunks = split_into_chunks(text, max_chunk_size=30, min_chunk_size=1)
    assert len(chunks) == 4
    assert all(len(chunk.split()) <= 30 for chunk in chunks)
