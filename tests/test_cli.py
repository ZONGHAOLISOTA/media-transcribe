from media_transcribe.cli import main


def test_missing_path_exits_2():
    assert main(["/no/such/path/xyz", "--dry-run"]) == 2


def test_unknown_engine_exits_2(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    assert main([str(tmp_path), "--engine", "nope", "--dry-run"]) == 2


def test_malformed_config_exits_2(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("engine: [unclosed\n", encoding="utf-8")
    assert main([str(tmp_path), "--config", str(bad), "--dry-run"]) == 2


def test_missing_context_file_exits_2(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    missing = tmp_path / "nope.txt"
    assert main([str(tmp_path), "--context-file", str(missing), "--dry-run"]) == 2


def test_unsupported_single_file_exits_2(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hi", encoding="utf-8")
    assert main([str(f), "--dry-run"]) == 2


def test_dry_run_succeeds(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    # dry-run does not require the engine to be installed (it warns, not fails).
    assert main([str(tmp_path), "--dry-run"]) == 0
