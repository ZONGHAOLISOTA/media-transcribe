import subprocess
import urllib.error

import pytest

from media_transcribe import doctor
from media_transcribe.cli import main
from media_transcribe.engines import hf_cache, qwen_mlx
from media_transcribe.engines.base import Check, Engine, Options, register

# `mlx-qwen3-asr --doctor` output captured from 0.4.4 (paths shortened).
UPSTREAM_OK = """\
mlx-qwen3-asr doctor (version 0.4.4)
python: 3.12.13
platform: darwin
[OK] ffmpeg: /opt/homebrew/bin/ffmpeg
[OK] mlx: installed
[OK] diarize extras: pyannote.audio + torch + torchcodec installed
[WARN] diarize auth token: not set
       needed for default pyannote/speaker-diarization-community-1
       fix: export PYANNOTE_AUTH_TOKEN=hf_...
doctor result: OK (1 warning(s))
"""
UPSTREAM_BROKEN = """\
[WARN] ffmpeg: not found on PATH (WAV input works without it)
       fix for mp3/m4a/mp4 and other formats: brew install ffmpeg
[FAIL] mlx: not installed
       fix: pip install mlx
[WARN] diarize extras: missing (optional)
       fix: pip install "mlx-qwen3-asr[diarize]"
[WARN] diarize auth token: not set
doctor result: FAIL (1 failure(s), 3 warning(s))
"""


def _upstream(monkeypatch, stdout):
    monkeypatch.setattr(qwen_mlx.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout, ""))


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
    check = qwen_mlx._diarization_check(True, (0, 4, 4))
    assert check.status == status
    if status == "fail":
        assert any(hint in check.detail for hint in ("http", "network"))  # a fix, not a shrug


@pytest.mark.parametrize("access", ["denied", "unauthorized", "error"])
def test_refused_token_with_cached_pipeline_only_warns(no_token_env, monkeypatch, access):
    snap = no_token_env / "models--pyannote--speaker-diarization-community-1" / "snapshots" / "r"
    snap.mkdir(parents=True)
    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    monkeypatch.setattr(hf_cache, "check_access", lambda repo, f, tok: (access, "why"))
    assert qwen_mlx._diarization_check(True, (0, 4, 4)).status == "warn"


def test_upstream_doctor_relay_when_healthy(monkeypatch):
    _upstream(monkeypatch, UPSTREAM_OK)
    checks = qwen_mlx._upstream_doctor(diarize=True)
    assert [(c.status, c.label) for c in checks] == [
        ("ok", "mlx-qwen3-asr: mlx"), ("ok", "mlx-qwen3-asr: diarize extras")]


@pytest.mark.parametrize("diarize,extras_status", [(False, "warn"), (True, "fail")])
def test_upstream_doctor_relay_when_broken(monkeypatch, diarize, extras_status):
    _upstream(monkeypatch, UPSTREAM_BROKEN)
    checks = {c.label: c for c in qwen_mlx._upstream_doctor(diarize=diarize)}
    assert set(checks) == {"mlx-qwen3-asr: mlx", "mlx-qwen3-asr: diarize extras"}  # no ffmpeg/token
    assert checks["mlx-qwen3-asr: mlx"].status == "fail"
    assert "uv tool install --reinstall mlx-qwen3-asr" in checks["mlx-qwen3-asr: mlx"].detail
    assert "pip" not in checks["mlx-qwen3-asr: mlx"].detail
    assert checks["mlx-qwen3-asr: diarize extras"].status == extras_status
    assert "mlx-qwen3-asr[diarize]" in checks["mlx-qwen3-asr: diarize extras"].detail


@pytest.mark.parametrize("which,version,status", [
    (None, None, "fail"),
    ("/bin/mlx-qwen3-asr", (0, 3, 2), "warn"),
    ("/bin/mlx-qwen3-asr", (0, 4, 4), "ok"),
])
def test_engine_version_gate(monkeypatch, which, version, status):
    monkeypatch.setattr(qwen_mlx.shutil, "which", lambda name: which)
    monkeypatch.setattr(qwen_mlx, "installed_version", lambda: version)
    monkeypatch.setattr(qwen_mlx, "_upstream_doctor", lambda diarize: [])
    monkeypatch.setattr(qwen_mlx, "_model_check", lambda *a, **k: Check("ok", "model"))
    monkeypatch.setattr(qwen_mlx, "_diarization_check", lambda *a: Check("ok", "diarization"))
    first = qwen_mlx.QwenMlxEngine().doctor(Options())[0]
    assert (first.label, first.status) == ("mlx-qwen3-asr", status)


def test_model_check_cases(no_token_env, tmp_path):
    (no_token_env / "models--Qwen--Qwen3-ASR-0.6B" / "snapshots" / "r").mkdir(parents=True)
    assert qwen_mlx._model_check(None).status == "ok"                       # cached default
    assert qwen_mlx._model_check("1.7b").status == "warn"                   # not downloaded
    assert qwen_mlx._model_check(str(tmp_path)).status == "ok"              # local folder
    missing = qwen_mlx._model_check("./no-such-model")
    assert missing.status == "fail" and "not found" in missing.detail


def test_doctor_flags_a_feature_the_engine_lacks(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: f"/bin/{name}")
    checks = doctor.collect("doctor-ok", Options(diarize=True))
    assert any(c.status == "fail" and "does not support: diarize" in c.detail for c in checks)


def test_cli_doctor_exit_codes(monkeypatch, capsys):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: f"/bin/{name}")
    assert main(["--doctor", "--engine", "doctor-ok"]) == 0
    assert main(["--doctor", "--engine", "doctor-crash"]) == 2
    assert "[FAIL]" in capsys.readouterr().out


def test_check_access_treats_a_redirect_as_access_without_following(monkeypatch):
    def redirect(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 302, "Found", {}, None)
    monkeypatch.setattr(hf_cache._OPENER, "open", redirect)
    assert hf_cache.check_access("org/repo", "config.yaml", "hf_abc") == ("ok", "")
    handler = hf_cache._NoRedirect()
    assert handler.redirect_request(None, None, 302, "Found", {}, "https://elsewhere") is None
