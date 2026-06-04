from pathlib import Path

import pytest

from media_transcribe.engines.base import Engine, available_engines, get_engine, register


class _FakeEngine(Engine):
    name = "fake-test"

    def is_available(self):
        return True, ""

    def transcribe(self, wav, *, language, context):
        return "fake text。"


def test_builtin_qwen_is_registered():
    assert "qwen-mlx" in available_engines()


def test_register_and_get():
    register(_FakeEngine())
    eng = get_engine("fake-test")
    assert eng.transcribe(Path("x.wav"), language="zh", context=None) == "fake text。"


def test_unknown_engine_raises():
    with pytest.raises(KeyError):
        get_engine("does-not-exist-xyz")
