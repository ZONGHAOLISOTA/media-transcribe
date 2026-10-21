import json
from pathlib import Path

from media_transcribe import batch
from media_transcribe.engines.base import Engine, Options, Segment


class _FakeEngine(Engine):
    name = "fake-batch"

    def is_available(self):
        return True, ""

    def transcribe(self, wav, *, language, context):
        return "hello。hello。world。"


class _FeatureEngine(Engine):
    name = "fake-features"
    features = frozenset({"model", "timestamps", "diarize"})

    def __init__(self):
        self.seen_options = []

    def is_available(self):
        return True, ""

    def transcribe(self, wav, *, language, context, options=None):
        self.seen_options.append(options)
        return "plain。"

    def transcribe_segments(self, wav, *, language, context, options):
        return [Segment("我们开始吧。", 0.0, 1.0, "S0"), Segment("好。", 1.2, 1.5, "S1"),
                Segment("好。", 1.5, 1.8, "S1")]


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_plan_mirrors_input_tree(tmp_path):
    _touch(tmp_path / "a" / "ep1.mp4")
    _touch(tmp_path / "b" / "ep1.mp4")
    files = batch.discover(tmp_path, recursive=True)
    items = batch.plan(tmp_path, files, tmp_path / "out")
    targets = {str(it.target.relative_to(tmp_path / "out")) for it in items}
    assert targets == {"a/ep1.txt", "b/ep1.txt"}
    assert batch.find_collisions(items) == {}


def test_collision_detected(tmp_path):
    _touch(tmp_path / "ep1.mp4")
    _touch(tmp_path / "ep1.mkv")  # both map to ep1.txt
    files = batch.discover(tmp_path, recursive=False)
    items = batch.plan(tmp_path, files, tmp_path / "out")
    assert len(batch.find_collisions(items)) == 1


def test_run_writes_transcript_then_skips(tmp_path, monkeypatch):
    _touch(tmp_path / "ep1.mp4")
    monkeypatch.setattr(batch.media, "extract_wav", lambda src, dst: dst)
    out = tmp_path / "out"

    results, code = batch.run(
        tmp_path, engine=_FakeEngine(), out_dir=out, language="zh",
        context_terms=[], recursive=False, overwrite=False, dry_run=False,
        log=lambda m: None,
    )
    assert code == 0
    txt = out / "ep1.txt"
    assert txt.read_text(encoding="utf-8").strip() == "hello。world。"

    manifest = json.loads((out / ".media-transcribe" / "run.json").read_text(encoding="utf-8"))
    assert manifest[0]["status"] == "done"

    results2, _ = batch.run(
        tmp_path, engine=_FakeEngine(), out_dir=out, language="zh",
        context_terms=[], recursive=False, overwrite=False, dry_run=False,
        log=lambda m: None,
    )
    assert results2[0].status == "skipped"


def _run(tmp_path, engine, overwrite=False, **kw):
    return batch.run(
        tmp_path, engine=engine, out_dir=tmp_path / "out", language="zh",
        context_terms=[], recursive=False, overwrite=overwrite, dry_run=False,
        log=lambda m: None, **kw,
    )


def test_diarized_run_writes_speaker_labels(tmp_path, monkeypatch):
    _touch(tmp_path / "ep1.mp4")
    monkeypatch.setattr(batch.media, "extract_wav", lambda src, dst: dst)
    results, code = _run(tmp_path, _FeatureEngine(), options=Options(diarize=True))
    assert code == 0
    text = (tmp_path / "out" / "ep1.txt").read_text(encoding="utf-8")
    assert text == "说话人 1：我们开始吧。\n\n说话人 2：好。\n"   # duplicate "好。" collapsed


def test_format_sets_extension(tmp_path, monkeypatch):
    _touch(tmp_path / "ep1.mp4")
    monkeypatch.setattr(batch.media, "extract_wav", lambda src, dst: dst)
    _run(tmp_path, _FeatureEngine(), fmt="srt")
    assert (tmp_path / "out" / "ep1.srt").read_text(encoding="utf-8").startswith("1\n00:00:00,000")


def test_options_reach_transcribe_only_when_requested(tmp_path, monkeypatch):
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "b.mp4")
    monkeypatch.setattr(batch.media, "extract_wav", lambda src, dst: dst)
    eng = _FeatureEngine()
    _run(tmp_path, eng)                                    # nothing requested
    _run(tmp_path, eng, options=Options(model="1.7b"), overwrite=True)
    assert eng.seen_options[0] is None
    assert eng.seen_options[-1].model == "1.7b"


def test_dry_run_does_not_write(tmp_path, monkeypatch):
    _touch(tmp_path / "ep1.mp4")
    monkeypatch.setattr(batch.media, "extract_wav", lambda src, dst: dst)
    out = tmp_path / "out"
    results, code = batch.run(
        tmp_path, engine=_FakeEngine(), out_dir=out, language="zh",
        context_terms=[], recursive=False, overwrite=False, dry_run=True,
        log=lambda m: None,
    )
    assert code == 0
    assert results[0].status == "planned"
    assert not (out / "ep1.txt").exists()
