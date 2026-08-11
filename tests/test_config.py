import pytest


def test_validate_config_requires_bot_token(monkeypatch):
    import config

    monkeypatch.setattr(config, "BOT_TOKEN", "")
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    with pytest.raises(RuntimeError):
        config.validate_config()
