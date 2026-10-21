import subprocess
import sys
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


def _cache_pipeline(root, repo="pyannote/speaker-diarization-community-1", config=True):
    snap = root / ("models--" + repo.replace("/", "--")) / "snapshots" / "r"
    snap.mkdir(parents=True)
    if config:
        (snap / "config.yaml").write_text("x", encoding="utf-8")
    return snap


@pytest.mark.parametrize("access", ["denied", "unauthorized", "error"])
def test_refused_token_with_cached_pipeline_only_warns(no_token_env, monkeypatch, access):
    _cache_pipeline(no_token_env)
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
    assert "uv tool upgrade --reinstall mlx-qwen3-asr" in checks["mlx-qwen3-asr: mlx"].detail
    assert "pip" not in checks["mlx-qwen3-asr: mlx"].detail
    assert checks["mlx-qwen3-asr: diarize extras"].status == extras_status
    assert "mlx-qwen3-asr[diarize]" in checks["mlx-qwen3-asr: diarize extras"].detail


@pytest.mark.parametrize("which,probe,status", [
    (None, (None, ""), "fail"),                                   # not installed
    ("/bin/mlx-qwen3-asr", (None, "ImportError: numpy"), "fail"),  # crashes at start-up
    ("/bin/mlx-qwen3-asr", ((0, 3, 2), ""), "warn"),
    ("/bin/mlx-qwen3-asr", ((0, 4, 4), ""), "ok"),
])
def test_engine_version_gate(monkeypatch, which, probe, status):
    monkeypatch.setattr(qwen_mlx.shutil, "which", lambda name: which)
    monkeypatch.setattr(qwen_mlx, "_probe", lambda: probe)
    monkeypatch.setattr(qwen_mlx, "_upstream_doctor", lambda diarize: [])
    monkeypatch.setattr(qwen_mlx, "_model_check", lambda *a, **k: Check("ok", "model"))
    monkeypatch.setattr(qwen_mlx, "_diarization_check", lambda *a: Check("ok", "diarization"))
    first = qwen_mlx.QwenMlxEngine().doctor(Options())[0]
    assert (first.label, first.status) == ("mlx-qwen3-asr", status)


def test_probe_reads_only_its_own_version(monkeypatch):
    qwen_mlx._probe.cache_clear()
    crash = subprocess.CompletedProcess([], 1, "", "Traceback ...\nnumpy 2.2.6 broke\n")
    monkeypatch.setattr(qwen_mlx.subprocess, "run", lambda cmd, **kw: crash)
    assert qwen_mlx._probe() == (None, "numpy 2.2.6 broke")
    qwen_mlx._probe.cache_clear()
    ok = subprocess.CompletedProcess([], 0, "mlx-qwen3-asr 0.4.4\n", "")
    monkeypatch.setattr(qwen_mlx.subprocess, "run", lambda cmd, **kw: ok)
    assert qwen_mlx._probe() == ((0, 4, 4), "")
    qwen_mlx._probe.cache_clear()


def test_model_check_cases(no_token_env, tmp_path, monkeypatch):
    (no_token_env / "models--Qwen--Qwen3-ASR-0.6B" / "snapshots" / "r").mkdir(parents=True)
    assert qwen_mlx._model_check(None).status == "ok"                       # cached default
    assert qwen_mlx._model_check("1.7b").status == "warn"                   # not downloaded
    model = tmp_path / "models" / "qwen"
    model.mkdir(parents=True)
    assert qwen_mlx._model_check(str(model)).status == "fail"               # no config.json
    (model / "config.json").write_text("{}", encoding="utf-8")
    assert qwen_mlx._model_check(str(model)).status == "ok"                 # local folder
    monkeypatch.chdir(tmp_path)
    assert qwen_mlx._model_check("models/qwen").status == "ok"              # relative folder
    missing = qwen_mlx._model_check("./no-such-model")
    assert missing.status == "fail" and "not found" in missing.detail
    typo = qwen_mlx._model_check("1.7b-8bits")
    assert typo.status == "fail" and "unknown model" in typo.detail


def test_import_check_really_imports(tmp_path):
    assert qwen_mlx._import_check(sys.executable, "json", "runtime", "fix it").status == "ok"
    bad = qwen_mlx._import_check(sys.executable, "no_such_module_xyz", "runtime", "fix it")
    assert bad.status == "fail" and "No module named" in bad.detail and "fix it" in bad.detail
    assert qwen_mlx._import_check(None, "json", "runtime", "fix").status == "warn"


def test_tool_python_comes_from_the_launcher_shebang(tmp_path):
    script = tmp_path / "mlx-qwen3-asr"
    script.write_text("#!/opt/tools/mlx/bin/python\nimport sys\n", encoding="utf-8")
    assert qwen_mlx._tool_python(str(script)) == "/opt/tools/mlx/bin/python"
    binary = tmp_path / "native"
    binary.write_bytes(b"\xcf\xfa\xed\xfe\x07\x00\x00\x01")
    assert qwen_mlx._tool_python(str(binary)) is None


def _quiet_engine_doctor(monkeypatch, imports):
    monkeypatch.setattr(qwen_mlx.shutil, "which", lambda name: "/bin/mlx-qwen3-asr")
    monkeypatch.setattr(qwen_mlx, "_probe", lambda: ((0, 4, 4), ""))
    monkeypatch.setattr(qwen_mlx, "_upstream_doctor", lambda diarize: [])
    monkeypatch.setattr(qwen_mlx, "_tool_python", lambda path: "/tool/python")
    monkeypatch.setattr(qwen_mlx, "_model_check", lambda *a, **k: Check("ok", "model"))
    monkeypatch.setattr(qwen_mlx, "_diarization_check", lambda *a: Check("ok", "diarization"))
    monkeypatch.setattr(qwen_mlx, "_import_check",
                        lambda python, module, label, fix: imports.append(module)
                        or Check("fail", label, module))


def test_diarize_doctor_tries_the_real_import(monkeypatch):
    # A present-but-broken pyannote (e.g. a binary wheel the OS refuses) must not pass.
    imports = []
    _quiet_engine_doctor(monkeypatch, imports)
    checks = qwen_mlx.QwenMlxEngine().doctor(Options(diarize=True))
    assert imports == ["pyannote.audio"]
    assert any(c.label == "diarization runtime" and c.status == "fail" for c in checks)


@pytest.mark.parametrize("language,timed,expected", [
    ("ja", True, ["nagisa"]), ("ko", True, ["soynlp"]), ("ja", False, []), ("zh", True, []),
])
def test_japanese_korean_alignment_extras_are_checked_for_timed_runs(
        monkeypatch, language, timed, expected):
    imports = []
    _quiet_engine_doctor(monkeypatch, imports)
    qwen_mlx.QwenMlxEngine().doctor(Options(), language=language, timed=timed)
    assert imports == expected


def test_core_doctor_passes_run_context_only_to_engines_that_take_it(monkeypatch):
    seen = []

    class _Ctx(_OkEngine):
        name = "doctor-ctx"

        def doctor(self, options, *, language=None, timed=False):
            seen.append((language, timed))
            return []

    register(_Ctx())
    monkeypatch.setattr(doctor.shutil, "which", lambda name: f"/bin/{name}")
    doctor.collect("doctor-ctx", Options(), language="ja", fmt="srt")
    assert seen == [("ja", True)]
    # An engine whose doctor() takes only options still reports its own error.
    assert doctor.collect("doctor-crash", Options(), language="ja")[-1].detail.endswith("boom")


def test_core_doctor_flags_a_timed_format_the_engine_cannot_do(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: f"/bin/{name}")
    checks = doctor.collect("doctor-ok", Options(), fmt="srt")
    assert any(c.status == "fail" and "does not support: timestamps" in c.detail for c in checks)


def test_interrupted_download_is_not_a_usable_cache(no_token_env):
    _cache_pipeline(no_token_env, config=False)        # snapshot without config.yaml
    assert qwen_mlx._diarization_check(True, (0, 4, 4)).status == "fail"


def test_custom_pipeline_id(no_token_env, monkeypatch):
    monkeypatch.setenv("PYANNOTE_MODEL_ID", "myorg/diarization")
    assert qwen_mlx._diarization_check(True, (0, 4, 4)).status == "warn"   # can't tell if gated
    _cache_pipeline(no_token_env, "myorg/diarization")
    assert qwen_mlx._diarization_check(True, (0, 4, 4)).status == "ok"


def test_local_pipeline_folder(no_token_env, monkeypatch, tmp_path):
    folder = tmp_path / "pipeline"
    folder.mkdir()
    monkeypatch.setenv("PYANNOTE_MODEL_ID", str(folder))
    assert qwen_mlx._diarization_check(True, (0, 4, 4)).status == "fail"   # no config.yaml
    (folder / "config.yaml").write_text("x", encoding="utf-8")
    assert qwen_mlx._diarization_check(True, (0, 4, 4)).status == "ok"


def test_refused_token_hint_names_the_variable_in_use(no_token_env, monkeypatch):
    monkeypatch.setenv("PYANNOTE_AUTH_TOKEN", "hf_bad")
    monkeypatch.setattr(hf_cache, "check_access", lambda repo, f, tok: ("unauthorized", "x"))
    detail = qwen_mlx._diarization_check(True, (0, 4, 4)).detail
    assert "export PYANNOTE_AUTH_TOKEN" in detail and "export HF_TOKEN" not in detail


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
