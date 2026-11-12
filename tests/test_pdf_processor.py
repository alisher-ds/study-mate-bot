import pytest
from pypdf import PdfWriter

from pdf_processor import _hard_split, extract_text_from_pdf, split_into_chunks

MAX = 60  # token limit used by most tests; overlap is 12% -> 7 tokens, 53 left for new text


def count(text):
    """Stub tokenizer: one token per 3 characters of every word (so "w12" = 1, "long123" = 3)."""
    return sum(-(-len(w) // 3) for w in text.split())


def chunk(text, max_tokens=MAX, **kwargs):
    return split_into_chunks(text, count, max_tokens, **kwargs)


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def shared_prefix_len(previous, following):
    """How many words at the end of `previous` repeat at the start of `following`."""
    a, b = previous.split(), following.split()
    for k in range(min(len(a), len(b)), 0, -1):
        if a[-k:] == b[:k]:
            return k
    return 0


def rebuild(chunks):
    """Glue chunks back together, dropping the overlap; inverse of chunking for unique words."""
    result = chunks[0].split()
    for following in chunks[1:]:
        k = shared_prefix_len(" ".join(result), following)
        result.extend(following.split()[k:])
    return result


# ---------------------------------------------------------------- basics

def test_empty_text_gives_no_chunks():
    assert chunk("") == []
    assert chunk("  \n\n \n\n  ") == []


def test_short_text_is_a_single_chunk():
    assert chunk("Salom dunyo.") == ["Salom dunyo."]


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        chunk("text", max_tokens=1)
    with pytest.raises(ValueError):
        chunk("text", overlap_ratio=0.6)
    with pytest.raises(ValueError):
        chunk("text", overlap_ratio=-0.1)


# ---------------------------------------------------------------- token limit

MIXED_TEXT = "\n\n".join([
    words(10, "a"),
    ". ".join(f"Gap raqami {i} shu yerda" for i in range(80)) + ".",  # long, with sentences
    words(300, "u"),  # long, no sentence ends
    "x" * 400,  # one huge "word" (URL / base64 like)
    words(30, "b"),
    "Oxirgi qisqa paragraf.",
])


@pytest.mark.parametrize("limit", [8, 20, MAX, 118])
def test_no_chunk_exceeds_the_token_limit(limit):
    chunks = chunk(MIXED_TEXT, max_tokens=limit)
    assert chunks
    assert all(count(c) <= limit for c in chunks)


def test_limit_holds_for_a_tokenizer_that_is_not_additive():
    # Counting the whole text costs more than the sum of its parts, so the packing estimate
    # is too low; the final check has to split such chunks instead of emitting them.
    def uneven(text):
        return count(text) + len(text) // 25

    chunks = split_into_chunks(MIXED_TEXT, uneven, MAX)
    assert all(uneven(c) <= MAX for c in chunks)
    original = {w for w in MIXED_TEXT.split()}
    assert original <= {w for c in chunks for w in c.split()} | {"x" * 400}


def test_oversized_word_is_split_within_the_limit_and_not_lost():
    chunks = chunk("x" * 500)
    assert all(count(c) <= MAX for c in chunks)
    assert sum(c.count("x") for c in chunks) >= 500


def test_oversized_word_is_cut_into_pieces_that_fit_before_packing():
    # Cut up front, not by the last-resort check, so no chunk is left holding only an overlap.
    parts = _hard_split("x" * 500, 53, count)
    assert "".join(parts) == "x" * 500
    assert all(count(p) <= 53 for p in parts)
    assert len(parts) <= 4


# ---------------------------------------------------------------- paragraphs and sentences

def test_paragraphs_are_merged_until_the_limit():
    # 25 tokens each: two fit into the 53 tokens of new text, the third does not.
    a, b, c = words(25, "a"), words(25, "b"), words(25, "c")
    chunks = chunk(f"{a}\n\n{b}\n\n{c}")

    assert len(chunks) == 2
    assert chunks[0] == f"{a}\n\n{b}"
    assert chunks[1].endswith(c)


def test_small_paragraphs_are_never_cut_in_the_middle():
    paragraphs = [f"{words(12, f'p{i}x')}." for i in range(30)]
    chunks = chunk("\n\n".join(paragraphs))

    assert len(chunks) > 1
    for paragraph in paragraphs:
        assert any(paragraph in c for c in chunks)


def test_long_paragraph_is_split_at_sentence_boundaries():
    paragraph = " ".join(f"Bu {i} raqamli gap." for i in range(60))
    chunks = chunk(paragraph)

    assert len(chunks) > 1
    assert all(c.endswith(".") for c in chunks)


def test_long_paragraph_without_sentences_is_split_by_words_without_loss():
    original = words(400)
    chunks = chunk(original)

    assert len(chunks) > 1
    assert rebuild(chunks) == original.split()


def test_very_long_paragraph_does_not_drop_previous_text():
    previous = words(20, "prev")
    long_para = words(700, "long")
    chunks = chunk(f"{previous}\n\n{long_para}")

    assert chunks[0] == previous
    assert rebuild(chunks) == (previous + " " + long_para).split()


def test_text_after_a_long_paragraph_is_kept():
    chunks = chunk(f"{words(700, 'long')}\n\ntail paragraph")
    assert chunks[-1].endswith("tail paragraph")


# ---------------------------------------------------------------- overlap

def test_consecutive_chunks_overlap_by_about_twelve_percent():
    limit = 100
    chunks = chunk(words(600), max_tokens=limit)
    assert len(chunks) > 3

    for previous, following in zip(chunks, chunks[1:], strict=False):
        k = shared_prefix_len(previous, following)
        shared_tokens = count(" ".join(following.split()[:k]))
        assert 0.10 * limit <= shared_tokens <= 0.15 * limit


def test_overlap_can_be_turned_off():
    chunks = chunk(words(400), overlap_ratio=0)
    assert len(chunks) > 1
    assert all(shared_prefix_len(a, b) == 0 for a, b in zip(chunks, chunks[1:], strict=False))
    assert " ".join(chunks).split() == words(400).split()


def test_overlap_comes_from_the_end_of_the_previous_chunk():
    a, b = words(40, "a"), words(40, "b")
    chunks = chunk(f"{a}\n\n{b}")

    assert len(chunks) == 2
    tail = chunks[1].split()[: shared_prefix_len(chunks[0], chunks[1])]
    assert tail and chunks[0].split()[-len(tail):] == tail


# ---------------------------------------------------------------- PDF text layer

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
    assert chunk(text) == []
