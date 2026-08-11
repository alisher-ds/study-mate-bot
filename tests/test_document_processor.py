from document_processor import extract_text, split_into_chunks


def test_split_chunks_preserves_text():
    text = "Birinchi paragraf.\n\nIkkinchi paragraf."
    chunks = split_into_chunks(text, max_words=20, min_words=1)
    assert len(chunks) == 1
    assert "Birinchi paragraf." in chunks[0]
    assert "Ikkinchi paragraf." in chunks[0]


def test_txt_extraction(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Machine Learning asoslari", encoding="utf-8")
    assert extract_text(str(path)) == "Machine Learning asoslari"


def test_unsupported_format(tmp_path):
    path = tmp_path / "file.exe"
    path.write_bytes(b"test")
    try:
        extract_text(str(path))
    except ValueError as exc:
        assert "Qo'llab-quvvatlanmaydigan" in str(exc)
    else:
        raise AssertionError("Unsupported file should raise ValueError")
