import json
from pathlib import Path

from media_transcribe import batch
from media_transcribe.engines.base import Engine


class _FakeEngine(Engine):
    name = "fake-batch"

    def is_available(self):
        return True, ""

    def transcribe(self, wav, *, language, context):
        return "hello。hello。world。"


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
