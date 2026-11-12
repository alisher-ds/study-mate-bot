import asyncio
import logging
import os
from datetime import datetime
from types import SimpleNamespace

import pytest
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.storage.base import StorageKey

import handlers

CHAT_ID = USER_ID = 7


# ---------------------------------------------------------------- registration order

def registered():
    return list(handlers.router.message.handlers)


def command_names(handler):
    names = []
    for f in handler.filters:
        if isinstance(f.callback, Command):
            names.extend(f.callback.commands)
    return names


def test_command_handlers_are_registered_before_the_free_text_handler():
    order = [h.callback.__name__ for h in registered()]
    question_index = order.index("handle_question")

    commands = {name: i for i, h in enumerate(registered()) for name in command_names(h)}
    assert {"start", "help", "reset", "test", "xulosa"} <= set(commands)
    assert all(index < question_index for index in commands.values())


def test_free_text_handler_is_last():
    assert registered()[-1].callback.__name__ == "handle_question"


# ---------------------------------------------------------------- dispatcher harness

@pytest.fixture(scope="module")
def dispatcher():
    dp = Dispatcher()
    dp.include_router(handlers.router)
    return dp


@pytest.fixture
def bot():
    return Bot(token="123456:TEST-token")


@pytest.fixture
def sent(monkeypatch):
    """Collects every Message.answer() text instead of calling Telegram."""
    texts = []

    async def fake_answer(self, text, **kwargs):
        texts.append(text)

    monkeypatch.setattr(types.Message, "answer", fake_answer)
    return texts


def make_message(text=None, document=None, contact=None):
    entities = None
    if text and text.startswith("/"):
        entities = [types.MessageEntity(type="bot_command", offset=0, length=len(text.split()[0]))]
    return types.Message(
        message_id=1,
        date=datetime.now(),
        chat=types.Chat(id=CHAT_ID, type="private"),
        from_user=types.User(id=USER_ID, is_bot=False, first_name="Test"),
        text=text,
        entities=entities,
        document=document,
        contact=contact,
    )


def feed(dispatcher, bot, message, state=None):
    async def go():
        if state is not None:
            key = StorageKey(bot_id=bot.id, chat_id=CHAT_ID, user_id=USER_ID)
            await dispatcher.storage.set_state(key, state)
        try:
            await dispatcher.feed_update(bot, types.Update(update_id=1, message=message))
        finally:
            key = StorageKey(bot_id=bot.id, chat_id=CHAT_ID, user_id=USER_ID)
            await dispatcher.storage.set_state(key, None)

    asyncio.run(go())


def stub_rag(monkeypatch):
    calls = []

    def make(name, result):
        async def fake(*args, **kwargs):
            calls.append(name)
            if isinstance(result, Exception):
                raise result
            return result
        monkeypatch.setattr(handlers, name, fake)

    make("generate_quiz", "QUIZ")
    make("generate_summary", "SUMMARY")
    make("generate_answer", "ANSWER")
    make("search_relevant_chunks", [])
    make("reset_user_data", True)
    return calls, make


# ---------------------------------------------------------------- routing behaviour

def test_test_command_reaches_cmd_test_not_the_question_handler(dispatcher, bot, sent, monkeypatch):
    calls, _ = stub_rag(monkeypatch)
    feed(dispatcher, bot, make_message("/test"))
    assert calls == ["generate_quiz"]
    assert "QUIZ" in sent


def test_xulosa_command_reaches_cmd_summary(dispatcher, bot, sent, monkeypatch):
    calls, _ = stub_rag(monkeypatch)
    feed(dispatcher, bot, make_message("/xulosa"))
    assert calls == ["generate_summary"]
    assert "SUMMARY" in sent


def test_help_and_reset_commands(dispatcher, bot, sent, monkeypatch):
    calls, _ = stub_rag(monkeypatch)
    feed(dispatcher, bot, make_message("/help"))
    feed(dispatcher, bot, make_message("/reset"))
    assert handlers.HELP_TEXT in sent
    assert calls == ["reset_user_data"]


def test_plain_text_is_answered_by_the_question_handler(dispatcher, bot, sent, monkeypatch):
    calls, _ = stub_rag(monkeypatch)
    feed(dispatcher, bot, make_message("Fotosintez nima?"))
    assert calls == ["search_relevant_chunks", "generate_answer"]
    assert "ANSWER" in sent


def test_raw_exception_text_is_not_shown_to_the_user(dispatcher, bot, sent, monkeypatch, caplog):
    _, make = stub_rag(monkeypatch)
    make("generate_answer", RuntimeError("secret-internal-detail"))
    with caplog.at_level(logging.ERROR):
        feed(dispatcher, bot, make_message("savol"))

    assert handlers.GENERIC_ERROR in sent
    assert not any("secret-internal-detail" in text for text in sent)
    assert "secret-internal-detail" in caplog.text  # ...but it is logged


# ---------------------------------------------------------------- registration

def test_text_during_phone_step_gets_a_helpful_reply(dispatcher, bot, sent):
    feed(dispatcher, bot, make_message("+998901234567"), state=handlers.Registration.telefon)
    assert len(sent) == 1
    assert "tugma" in sent[0]


def test_text_during_phone_step_does_not_reach_the_question_handler(
    dispatcher, bot, sent, monkeypatch
):
    calls, _ = stub_rag(monkeypatch)
    feed(dispatcher, bot, make_message("hello"), state=handlers.Registration.telefon)
    assert calls == []


def test_someone_elses_contact_is_rejected(dispatcher, bot, sent):
    contact = types.Contact(phone_number="+998900000000", first_name="Other", user_id=USER_ID + 1)
    feed(dispatcher, bot, make_message(contact=contact), state=handlers.Registration.telefon)
    assert len(sent) == 1
    assert "o'zingizning" in sent[0]


# ---------------------------------------------------------------- PDF upload

def document(file_name, mime_type=None, file_size=1000):
    return types.Document(
        file_id="f", file_unique_id="u", file_name=file_name, mime_type=mime_type,
        file_size=file_size,
    )


@pytest.fixture
def pdf_env(monkeypatch):
    """Fakes Telegram download + PDF parsing; records where the file was written."""
    env = SimpleNamespace(paths=[], text="Some extracted text.\n\nMore text.", stored=[])

    async def fake_get_file(self, file_id):
        return SimpleNamespace(file_path="remote/path")

    async def fake_download(self, file_path, destination):
        env.paths.append(destination)
        with open(destination, "wb") as fh:
            fh.write(b"%PDF-fake")

    async def fake_add_document(user_id, chunks, doc_name):
        env.stored.append((user_id, chunks, doc_name))
        return len(chunks)

    monkeypatch.setattr(Bot, "get_file", fake_get_file)
    monkeypatch.setattr(Bot, "download_file", fake_download)
    monkeypatch.setattr(handlers, "extract_text_from_pdf", lambda path: env.text)
    monkeypatch.setattr(handlers, "add_document", fake_add_document)
    return env


@pytest.mark.parametrize("name", ["notes.pdf", "NOTES.PDF", "Lecture.Pdf"])
def test_pdf_extension_check_is_case_insensitive(dispatcher, bot, sent, pdf_env, name):
    feed(dispatcher, bot, make_message(document=document(name)))
    assert pdf_env.stored and pdf_env.stored[0][2] == name
    assert any("muvaffaqiyatli" in text for text in sent)


def test_missing_file_name_is_accepted_when_mime_type_is_pdf(dispatcher, bot, sent, pdf_env):
    feed(dispatcher, bot, make_message(document=document(None, "application/pdf")))
    assert pdf_env.stored


def test_missing_file_name_and_non_pdf_mime_is_rejected(dispatcher, bot, sent, pdf_env):
    feed(dispatcher, bot, make_message(document=document(None, "image/png")))
    assert not pdf_env.stored
    assert any("faqat PDF" in text for text in sent)


def test_non_pdf_document_is_rejected(dispatcher, bot, sent, pdf_env):
    feed(dispatcher, bot, make_message(document=document("photo.png", "image/png")))
    assert not pdf_env.stored


def test_oversized_pdf_is_rejected_before_download(dispatcher, bot, sent, pdf_env):
    big = document("big.pdf", file_size=handlers.MAX_PDF_BYTES + 1)
    feed(dispatcher, bot, make_message(document=big))
    assert pdf_env.paths == []
    assert any("katta" in text for text in sent)


def test_download_path_ignores_the_user_file_name_and_is_cleaned_up(
    dispatcher, bot, sent, pdf_env
):
    feed(dispatcher, bot, make_message(document=document("../../evil name.pdf")))

    (path,) = pdf_env.paths
    assert "evil" not in path
    assert os.path.commonpath([path, os.getcwd()]) != os.getcwd()  # not inside the project
    assert not os.path.exists(path)
    assert not os.path.exists(os.path.dirname(path))


def test_scanned_pdf_gets_a_clear_message_not_zero_chunks(dispatcher, bot, sent, pdf_env):
    pdf_env.text = ""
    feed(dispatcher, bot, make_message(document=document("scan.pdf")))

    assert pdf_env.stored == []
    assert any("skanerlangan" in text for text in sent)
    assert not any("muvaffaqiyatli" in text or "0 ta" in text for text in sent)


def test_pdf_failure_shows_a_friendly_message(dispatcher, bot, sent, pdf_env, monkeypatch, caplog):
    def boom(path):
        raise RuntimeError("secret-parser-detail")

    monkeypatch.setattr(handlers, "extract_text_from_pdf", boom)
    with caplog.at_level(logging.ERROR):
        feed(dispatcher, bot, make_message(document=document("a.pdf")))

    assert handlers.GENERIC_ERROR in sent
    assert not any("secret-parser-detail" in text for text in sent)
    assert "secret-parser-detail" in caplog.text
