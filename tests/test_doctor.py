import pytest

from media_transcribe import doctor
from media_transcribe.engines import hf_cache, qwen_mlx
from media_transcribe.engines.base import Check, Engine, Options, register


class _OkEngine(Engine):
    name = "doctor-ok"

    def is_available(self):
        return True, ""

    def transcribe(self, wav, *, language, context):
        return ""


class _CrashingEngine(_OkEngine):
    name = "doctor-crash"

    def doctor(self, options):
        raise RuntimeError("boom")


register(_OkEngine())
register(_CrashingEngine())


def test_all_ok_exits_0(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: f"/bin/{name}")
    lines = []
    assert doctor.run("doctor-ok", Options(), log=lines.append) == 0
    assert any(line.startswith("[OK]") and "ffmpeg" in line for line in lines)


def test_missing_ffmpeg_fails_with_a_fix(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    lines = []
    assert doctor.run("doctor-ok", Options(), log=lines.append) == 2
    assert any("[FAIL]" in line and "brew install ffmpeg" in line for line in lines)


def test_unknown_or_crashing_engine_is_reported(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: f"/bin/{name}")
    assert doctor.collect("no-such-engine", Options())[-1].status == "fail"
    checks = doctor.collect("doctor-crash", Options())
    assert checks[-1] == Check("fail", "engine doctor-crash", "doctor check crashed: boom")


def test_crash_message_never_shows_a_token(monkeypatch):
    class _Leaky(_OkEngine):
        name = "doctor-leaky"

        def doctor(self, options):
            raise ValueError("Invalid header value b'Bearer hf_SECRETSECRET'")

    register(_Leaky())
    monkeypatch.setattr(doctor.shutil, "which", lambda name: f"/bin/{name}")
    detail = doctor.collect("doctor-leaky", Options())[-1].detail
    assert "SECRETSECRET" not in detail and "hf_***" in detail


@pytest.fixture
def no_token_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "no-token-file"))
    for var in (*hf_cache.TOKEN_VARS, "PYANNOTE_MODEL_ID"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def test_diarization_check_without_token_or_cache(no_token_env):
    soft = qwen_mlx._diarization_check(False, (0, 4, 4))
    hard = qwen_mlx._diarization_check(True, (0, 4, 4))
    assert (soft.status, hard.status) == ("warn", "fail")
    assert "huggingface.co/pyannote/speaker-diarization-community-1" in hard.detail


def test_diarization_check_cached_needs_no_token(no_token_env):
    snap = no_token_env / "models--pyannote--speaker-diarization-community-1" / "snapshots" / "r1"
    snap.mkdir(parents=True)
    (snap / "config.yaml").write_text("x", encoding="utf-8")
    check = qwen_mlx._diarization_check(True, (0, 4, 4))
    assert check.status == "ok" and "no token needed" in check.detail


@pytest.mark.parametrize("access,status", [("ok", "ok"), ("denied", "fail"),
                                           ("unauthorized", "fail"), ("error", "fail")])
def test_diarization_check_with_token(no_token_env, monkeypatch, access, status):
    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    monkeypatch.setattr(hf_cache, "check_access", lambda repo, f, tok: (access, "why"))
    assert qwen_mlx._diarization_check(True, (0, 4, 4)).status == status
