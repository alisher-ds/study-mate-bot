from pypdf import PdfWriter

from pdf_processor import extract_text_from_pdf, split_into_chunks


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_empty_text_gives_no_chunks():
    assert split_into_chunks("") == []
    assert split_into_chunks("  \n\n \n\n  ") == []


def test_short_text_is_a_single_chunk():
    assert split_into_chunks("Salom dunyo.") == ["Salom dunyo."]


def test_paragraphs_are_merged_up_to_max_size():
    text = "\n\n".join([words(60, "a"), words(60, "b"), words(60, "c")])
    chunks = split_into_chunks(text, max_chunk_size=130, min_chunk_size=50)
    assert len(chunks) == 2
    assert chunks[0].startswith("a0") and "b59" in chunks[0]
    assert chunks[1].startswith("c0")


def test_very_long_paragraph_does_not_drop_previous_text():
    # Fewer words than min_chunk_size: the old code kept it in current_chunk and then
    # threw it away when the long paragraph was split.
    previous = words(20, "prev")
    long_para = words(700, "long")
    chunks = split_into_chunks(f"{previous}\n\n{long_para}", max_chunk_size=100, min_chunk_size=50)

    assert chunks[0] == previous
    assert all(len(c.split()) <= 100 for c in chunks)
    # nothing lost, nothing duplicated
    assert " ".join(chunks).split() == (previous + " " + long_para).split()


def test_very_long_paragraph_flushes_a_full_previous_chunk():
    previous = words(70, "prev")
    chunks = split_into_chunks(f"{previous}\n\n{words(700, 'long')}", max_chunk_size=100)
    assert chunks[0] == previous


def test_very_long_paragraph_alone_is_split_by_words():
    chunks = split_into_chunks(words(450), max_chunk_size=100)
    assert [len(c.split()) for c in chunks] == [100, 100, 100, 100, 50]


def test_text_after_long_paragraph_starts_a_new_chunk():
    chunks = split_into_chunks(f"{words(700, 'long')}\n\ntail paragraph", max_chunk_size=100)
    assert chunks[-1] == "tail paragraph"


def test_blank_pdf_pages_yield_empty_text(tmp_path):
    # A PDF with only blank pages behaves like a scanned PDF: no text layer.
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    path = tmp_path / "scan.pdf"
    with open(path, "wb") as fh:
        writer.write(fh)

    text = extract_text_from_pdf(str(path))

    assert text == ""
    assert split_into_chunks(text) == []
