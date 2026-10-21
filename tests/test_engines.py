import json
import subprocess
from pathlib import Path

import pytest

from media_transcribe.engines import hf_cache, qwen_mlx
from media_transcribe.engines.base import (
    Engine, Options, available_engines, get_engine, register,
)


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


def test_options_report_requested_features():
    assert Options().requested_features() == set()
    assert Options(model="1.7b", diarize=True).requested_features() == {"model", "diarize"}


@pytest.mark.parametrize("given,expected", [
    (None, "Qwen/Qwen3-ASR-0.6B"),
    ("1.7b", "Qwen/Qwen3-ASR-1.7B"),
    ("1.7B-8bit", "moona3k/mlx-qwen3-asr-1.7b-8bit"),
    ("someone/custom-asr", "someone/custom-asr"),
])
def test_model_aliases(given, expected):
    assert qwen_mlx.resolve_model(given) == expected


class _FakeRun:
    """Stands in for subprocess.run; records commands, fakes mlx-qwen3-asr output."""

    def __init__(self, stdout="转写结果。", payload=None):
        self.calls, self.envs = [], []
        self.stdout, self.payload = stdout, payload

    def __call__(self, cmd, capture_output, text, env=None, **kw):
        self.calls.append(cmd)
        self.envs.append(env)
        if self.payload is not None and "--output-dir" in cmd:
            out = Path(cmd[cmd.index("--output-dir") + 1]) / (Path(cmd[-1]).stem + ".json")
            out.write_text(json.dumps(self.payload), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout=self.stdout, stderr="")


def test_plain_transcribe_pins_default_model(monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(qwen_mlx.subprocess, "run", fake)
    out = qwen_mlx.QwenMlxEngine().transcribe(Path("a.wav"), language="zh", context=["数据"])
    assert out == "转写结果。"
    cmd = fake.calls[0]
    assert cmd[cmd.index("--model") + 1] == "Qwen/Qwen3-ASR-0.6B"
    assert cmd[cmd.index("--language") + 1] == "Chinese"
    assert "--stdout-only" in cmd and cmd[cmd.index("--context") + 1] == "数据"


def test_segments_mode_builds_speaker_segments(monkeypatch, tmp_path):
    payload = {
        "text": "你好。你好呀。",
        "segments": [
            {"text": "你", "start": 0.0, "end": 0.3, "speaker": "SPEAKER_00"},
            {"text": "好", "start": 0.3, "end": 0.8, "speaker": "SPEAKER_00"},
            {"text": "你", "start": 1.0, "end": 1.3, "speaker": "SPEAKER_01"},
            {"text": "好", "start": 1.3, "end": 1.6, "speaker": "SPEAKER_01"},
            {"text": "呀", "start": 1.6, "end": 1.9, "speaker": "SPEAKER_01"},
        ],
    }
    fake = _FakeRun(payload=payload)
    monkeypatch.setattr(qwen_mlx.subprocess, "run", fake)
    monkeypatch.setattr(qwen_mlx, "_diarization_env", lambda: None)
    opts = Options(model="1.7b", diarize=True, num_speakers=2)
    segs = qwen_mlx.QwenMlxEngine().transcribe_segments(
        tmp_path / "a.wav", language="zh", context=None, options=opts)
    assert [(s.speaker, s.text) for s in segs] == [("SPEAKER_00", "你好。"),
                                                   ("SPEAKER_01", "你好呀。")]
    cmd = fake.calls[0]
    for flag in ("--timestamps", "--diarize"):
        assert flag in cmd
    assert cmd[cmd.index("--num-speakers") + 1] == "2"
    assert cmd[cmd.index("--model") + 1] == "Qwen/Qwen3-ASR-1.7B"
    assert "--min-speakers" not in cmd


def test_timed_run_without_word_timings_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.setattr(qwen_mlx.subprocess, "run", _FakeRun(payload={"text": "只有文字。"}))
    with pytest.raises(RuntimeError, match="no word timestamps"):
        qwen_mlx.QwenMlxEngine().transcribe_segments(
            tmp_path / "a.wav", language="zh", context=None, options=Options(timestamps=True))


def test_min_speakers_above_upstream_default_max_sends_a_max(monkeypatch, tmp_path):
    payload = {"text": "好。", "segments": [{"text": "好", "start": 0, "end": .5, "speaker": "S"}]}
    fake = _FakeRun(payload=payload)
    monkeypatch.setattr(qwen_mlx.subprocess, "run", fake)
    monkeypatch.setattr(qwen_mlx, "_diarization_env", lambda: None)
    qwen_mlx.QwenMlxEngine().transcribe_segments(
        tmp_path / "a.wav", language="zh", context=None,
        options=Options(diarize=True, min_speakers=9))
    cmd = fake.calls[0]
    assert cmd[cmd.index("--max-speakers") + 1] == "9"


def test_home_relative_model_path_is_expanded():
    assert qwen_mlx.resolve_model("~/models/asr") == str(Path("~/models/asr").expanduser())


def test_engine_failure_raises_with_stderr_tail(monkeypatch):
    def failing(cmd, capture_output, text, env=None, **kw):
        return subprocess.CompletedProcess(cmd, 3, stdout="", stderr="noise\nreal error here")
    monkeypatch.setattr(qwen_mlx.subprocess, "run", failing)
    with pytest.raises(RuntimeError, match="real error here"):
        qwen_mlx.QwenMlxEngine().transcribe(Path("a.wav"), language="en", context=None)


def _fake_cache(root: Path, repo: str) -> Path:
    base = root / ("models--" + repo.replace("/", "--"))
    snap = base / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.yaml").write_text("x", encoding="utf-8")
    (base / "refs").mkdir()
    (base / "refs" / "main").write_text("abc123", encoding="utf-8")
    return snap


def test_hf_cache_snapshot_and_token(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    snap = _fake_cache(tmp_path, "pyannote/speaker-diarization-community-1")
    assert hf_cache.snapshot("pyannote/speaker-diarization-community-1") == snap
    assert hf_cache.snapshot("nobody/missing") is None
    for var in hf_cache.TOKEN_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "no-token-file"))
    assert hf_cache.token() == ""
    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    assert hf_cache.token() == "hf_abc"


def test_token_from_hf_login_file(monkeypatch, tmp_path):
    for var in hf_cache.TOKEN_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("HF_TOKEN_PATH", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert hf_cache.token() == ""
    (tmp_path / "token").write_text("hf_fromlogin\n", encoding="utf-8")
    assert hf_cache.token() == "hf_fromlogin"


def test_malformed_token_is_rejected_without_echoing_it():
    status, detail = hf_cache.check_access("org/repo", "config.yaml", "hf_SECRET\nhf_two")
    assert status == "unauthorized" and "SECRET" not in detail


def test_cached_pipeline_is_used_without_a_token(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "no-token-file"))
    for var in (*hf_cache.TOKEN_VARS, "PYANNOTE_MODEL_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(qwen_mlx, "installed_version", lambda: (0, 4, 4))
    assert qwen_mlx._diarization_env() is None          # nothing cached yet
    snap = _fake_cache(tmp_path, qwen_mlx.PYANNOTE_PIPELINE)
    assert qwen_mlx._diarization_env()["PYANNOTE_MODEL_ID"] == str(snap)
    monkeypatch.setenv("HF_TOKEN", "hf_abc")            # a token means: let upstream download
    assert qwen_mlx._diarization_env() is None
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setattr(qwen_mlx, "installed_version", lambda: (0, 3, 2))
    assert qwen_mlx._diarization_env() is None          # pyannote 3.1 era: leave it alone
