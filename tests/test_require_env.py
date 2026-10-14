import pytest
from conftest import require_env


def test_require_env_set(monkeypatch):
    monkeypatch.setenv("X_RAG", "v")
    assert require_env("X_RAG") == "v"


def test_require_env_skips_by_default(monkeypatch):
    monkeypatch.delenv("X_RAG", raising=False)
    monkeypatch.delenv("RAG_REQUIRE_INTEGRATION", raising=False)
    with pytest.raises(pytest.skip.Exception):
        require_env("X_RAG")


def test_require_env_fails_when_required(monkeypatch):
    monkeypatch.delenv("X_RAG", raising=False)
    monkeypatch.setenv("RAG_REQUIRE_INTEGRATION", "1")
    with pytest.raises(pytest.fail.Exception):
        require_env("X_RAG")
