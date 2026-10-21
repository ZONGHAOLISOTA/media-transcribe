import pytest

from media_transcribe.cli import _build_options, build_parser, main
from media_transcribe.config import load_config
from media_transcribe.engines.base import Engine, register


class _PlainOnlyEngine(Engine):
    """A v0.1-style engine: no optional features, no ``options`` parameter."""

    name = "plain-only-test"

    def is_available(self):
        return True, ""

    def transcribe(self, wav, *, language, context):
        return "x。"


register(_PlainOnlyEngine())


def test_missing_path_exits_2():
    assert main(["/no/such/path/xyz", "--dry-run"]) == 2


def test_no_arguments_prints_examples_and_exits_2(capsys):
    assert main([]) == 2
    err = capsys.readouterr().err
    assert "examples:" in err and "--doctor" in err


def test_help_is_grouped_with_examples():
    text = build_parser().format_help()
    for heading in ("recognition:", "speakers (interviews, meetings):", "output:",
                    "setup and other:", "examples:"):
        assert heading in text


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


def test_dry_run_with_new_options_succeeds(tmp_path, capsys):
    (tmp_path / "a.mp4").write_bytes(b"x")
    argv = [str(tmp_path), "--dry-run", "--diarize", "--num-speakers", "2", "--timestamps",
            "--format", "srt", "--model", "1.7b"]
    assert main(argv) == 0
    assert "a.srt" in capsys.readouterr().out


@pytest.mark.parametrize("extra", [
    ["--num-speakers", "2"],                                   # needs --diarize
    ["--diarize", "--num-speakers", "0"],                      # must be >= 1
    ["--diarize", "--min-speakers", "4", "--max-speakers", "2"],
    ["--diarize", "--num-speakers", "2", "--max-speakers", "3"],
])
def test_bad_speaker_options_exit_2(tmp_path, extra):
    (tmp_path / "a.mp4").write_bytes(b"x")
    assert main([str(tmp_path), "--dry-run", *extra]) == 2


def test_unknown_format_is_rejected(tmp_path):
    with pytest.raises(SystemExit) as e:
        main([str(tmp_path), "--format", "docx"])
    assert e.value.code == 2


def test_bad_format_in_config_exits_2(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("format: docx\n", encoding="utf-8")
    assert main([str(tmp_path), "--config", str(cfg), "--dry-run"]) == 2


@pytest.mark.parametrize("extra", [["--diarize"], ["--model", "big"], ["--format", "vtt"]])
def test_feature_the_engine_lacks_exits_2(tmp_path, extra, capsys):
    (tmp_path / "a.mp4").write_bytes(b"x")
    assert main([str(tmp_path), "--engine", "plain-only-test", "--dry-run", *extra]) == 2
    assert "does not support" in capsys.readouterr().err


def test_plain_engine_without_options_still_works(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    assert main([str(tmp_path), "--engine", "plain-only-test", "--dry-run"]) == 0


def _options(tmp_path, yaml_text, argv=()):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    args = build_parser().parse_args([str(tmp_path), *argv])
    return _build_options(args, load_config(cfg))[0]


def test_config_speaker_count_waits_for_diarize(tmp_path):
    assert _options(tmp_path, "num_speakers: 2\n").num_speakers is None      # plain run OK
    assert _options(tmp_path, "num_speakers: 2\n", ["--diarize"]).num_speakers == 2


def test_cli_speaker_counts_replace_the_configs(tmp_path):
    opts = _options(tmp_path, "diarize: true\nmin_speakers: 2\n", ["--num-speakers", "3"])
    assert (opts.num_speakers, opts.min_speakers) == (3, None)


def test_quoted_booleans_in_config(tmp_path):
    assert _options(tmp_path, 'diarize: "false"\ntimestamps: "yes"\n').diarize is False
    assert _options(tmp_path, 'timestamps: "yes"\n').timestamps is True


@pytest.mark.parametrize("yaml_text", [
    "model: 1.7\n",
    "diarize: true\nnum_speakers: 2.9\n",
    "diarize: maybe\n",
])
def test_bad_config_values_exit_2(tmp_path, yaml_text):
    (tmp_path / "a.mp4").write_bytes(b"x")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    assert main([str(tmp_path), "--config", str(cfg), "--dry-run"]) == 2


def test_no_flags_switch_off_config_defaults(tmp_path):
    opts = _options(tmp_path, "diarize: true\ntimestamps: true\n",
                    ["--no-diarize", "--no-timestamps"])
    assert (opts.diarize, opts.timestamps) == (False, False)


def test_null_format_means_default(tmp_path, capsys):
    (tmp_path / "a.mp4").write_bytes(b"x")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("format: null\nrecursive: \"false\"\n", encoding="utf-8")
    assert main([str(tmp_path), "--config", str(cfg), "--dry-run"]) == 0
    assert "a.txt" in capsys.readouterr().out


@pytest.mark.parametrize("yaml_text", ["output_dir: 5\n", "context_file: [a, b]\n",
                                       "recursive: sometimes\n"])
def test_mistyped_config_paths_exit_2_without_traceback(tmp_path, yaml_text):
    (tmp_path / "a.mp4").write_bytes(b"x")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    assert main([str(tmp_path), "--config", str(cfg), "--dry-run"]) == 2


def test_non_utf8_context_file_exits_2(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    vocab = tmp_path / "vocab.txt"
    vocab.write_bytes("术语".encode("gbk"))
    assert main([str(tmp_path), "--context-file", str(vocab), "--dry-run"]) == 2


def test_null_language_means_the_default(tmp_path, capsys):
    (tmp_path / "a.mp4").write_bytes(b"x")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("language: null\n", encoding="utf-8")
    assert main([str(tmp_path), "--config", str(cfg), "--dry-run", "-v"]) == 0
    assert "lang=zh" in capsys.readouterr().err


@pytest.mark.parametrize("yaml_text", ["engine: [a, b]\n", "language: [zh, en]\n"])
def test_non_text_engine_or_language_exits_2(tmp_path, yaml_text):
    (tmp_path / "a.mp4").write_bytes(b"x")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    assert main([str(tmp_path), "--config", str(cfg), "--dry-run"]) == 2


def test_unwritable_output_folder_exits_2(tmp_path, capsys):
    (tmp_path / "a.mp4").write_bytes(b"x")
    out = tmp_path / "locked"
    out.mkdir()
    out.chmod(0o500)
    try:
        assert main([str(tmp_path), "--engine", "plain-only-test", "--out", str(out)]) == 2
    finally:
        out.chmod(0o700)
    assert "not writable" in capsys.readouterr().err


def test_verbose_prints_the_settings(tmp_path, capsys):
    (tmp_path / "a.mp4").write_bytes(b"x")
    assert main([str(tmp_path), "--dry-run", "-v", "--diarize", "--num-speakers", "2",
                 "--model", "1.7b"]) == 0
    err = capsys.readouterr().err
    assert "model=1.7b" in err and "diarize=on (num_speakers=2)" in err


def test_config_supplies_new_options(tmp_path, capsys):
    (tmp_path / "a.mp4").write_bytes(b"x")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("diarize: true\nnum_speakers: 2\nformat: json\n", encoding="utf-8")
    assert main([str(tmp_path), "--config", str(cfg), "--dry-run"]) == 0
    assert "a.json" in capsys.readouterr().out
