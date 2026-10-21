"""Command-line entry point for media-transcribe."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .engines.base import FEATURE_TIMESTAMPS, Options, available_engines, get_engine
from .media import is_media
from .render import FORMATS, TIMED_FORMATS

EXAMPLES = """\
examples:
  media-transcribe --doctor                            check the setup (start here)
  media-transcribe lecture.mp4                         one speaker (lecture, talk)
  media-transcribe interview.m4a --diarize             label who speaks (interview, meeting)
  media-transcribe meeting.m4a --diarize --timestamps  ...with [hh:mm:ss] times
  media-transcribe chat.m4a --diarize --num-speakers 2
                                                       fix the speaker count (only if sure)
  media-transcribe talk.mp4 --model 1.7b               more accurate, slower model
  media-transcribe talk.mp4 --format srt               subtitles
  media-transcribe ~/Videos --recursive --dry-run      preview a folder tree

Transcripts go to <input folder>/transcripts/ (or --out). Files that already have a
non-empty transcript are skipped; add --overwrite to redo them.
"""


def _resolve_context(context_file: str | None, context_inline: str | None) -> list[str]:
    """Merge file + inline domain vocab, de-duplicated, order-preserving."""
    terms: list[str] = []
    if context_file:
        terms += Path(context_file).read_text(encoding="utf-8").split()
    if context_inline:
        terms += context_inline.split()
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="media-transcribe",
        usage="media-transcribe <file-or-folder> [options]\n       media-transcribe --doctor",
        description="Transcribe video and audio files to text (engine-pluggable).",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    p.add_argument("path", type=Path, nargs="?", help="a media file, or a folder of them")

    rec = p.add_argument_group("recognition")
    rec.add_argument("--lang", dest="language", default=None, metavar="LANG",
                     help="spoken language: zh, en, ja, ko, ... (default: zh; "
                          "Chinese mixed with English: zh)")
    rec.add_argument("--model", default=None, metavar="MODEL",
                     help="0.6b (default, fast), 1.7b (more accurate, ~2x slower), "
                          "0.6b-8bit, 1.7b-8bit, 0.6b-4bit, 1.7b-4bit, "
                          "or a Hugging Face model id / local path")
    rec.add_argument("--context", dest="context", default=None, metavar='"TERMS"',
                     help="names / jargon to recognise, space-separated")
    rec.add_argument("--context-file", dest="context_file", default=None, metavar="FILE",
                     help="read such terms from a file (merged with --context)")
    rec.add_argument("--engine", default=None, metavar="NAME",
                     help=f"ASR engine (default: qwen-mlx; available: "
                          f"{', '.join(available_engines()) or 'none'})")

    spk = p.add_argument_group("speakers (interviews, meetings)")
    spk.add_argument("--diarize", action=argparse.BooleanOptionalAction, default=None,
                     help='label who speaks: "说话人 1：..." for zh, "Speaker 1: ..." '
                          "otherwise; needs a one-time setup, see --doctor")
    spk.add_argument("--num-speakers", type=int, default=None, metavar="N",
                     help="exact number of speakers; only if you are sure, a wrong "
                          "count hurts (default: detected)")
    spk.add_argument("--min-speakers", type=int, default=None, metavar="N",
                     help="at least N speakers when detecting")
    spk.add_argument("--max-speakers", type=int, default=None, metavar="N",
                     help="at most N speakers when detecting")

    out = p.add_argument_group("output")
    out.add_argument("--timestamps", action=argparse.BooleanOptionalAction, default=None,
                     help="start each line with its [hh:mm:ss] time")
    out.add_argument("--format", dest="fmt", default=None, choices=FORMATS, metavar="FMT",
                     help=f"{' | '.join(FORMATS)} (default: txt; srt/vtt/tsv are timed)")
    out.add_argument("--out", dest="out", type=Path, default=None, metavar="DIR",
                     help="output folder (default: <input folder>/transcripts)")
    out.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=None,
                     help="include subfolders")
    out.add_argument("--overwrite", action="store_true",
                     help="redo files that already have a transcript")

    oth = p.add_argument_group("setup and other")
    oth.add_argument("--doctor", action="store_true",
                     help="check ffmpeg, the engine, models and diarization setup, then exit")
    oth.add_argument("--dry-run", action="store_true",
                     help="list what would be transcribed, do nothing")
    oth.add_argument("--config", type=Path, default=None, metavar="FILE",
                     help="YAML file with defaults for these options "
                          "(see config.example.yaml)")
    oth.add_argument("-v", "--verbose", action="store_true",
                     help="also print the settings in effect")
    oth.add_argument("--version", action="version", version=f"media-transcribe {__version__}")
    oth.add_argument("-h", "--help", action="help", help="show this help and exit")
    return p


def _pick(flag, cfg_value):
    """CLI flag if given, else the config / default value."""
    return flag if flag is not None else cfg_value


_SPEAKER_KEYS = ("num_speakers", "min_speakers", "max_speakers")


def _flag(value, name: str) -> bool:
    """A yes/no setting; config files may spell it true/false, yes/no, on/off, 1/0."""
    if isinstance(value, bool) or value is None:
        return bool(value)
    word = str(value).strip().lower()
    if word in ("true", "yes", "on", "1"):
        return True
    if word in ("false", "no", "off", "0", ""):
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _speaker_count(value, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{name} must be a whole number, got {value!r}")
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number, got {value!r}") from None
    if n < 1:
        raise ValueError(f"{name} must be at least 1, got {n}")
    return n


def _build_options(args: argparse.Namespace, cfg: dict) -> tuple[Options, str]:
    """Resolve and validate the engine options and output format (raises ValueError)."""
    fmt = str(_pick(args.fmt, cfg["format"]) or "txt").lower()
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose one of {', '.join(FORMATS)}")
    model = _pick(args.model, cfg["model"])
    if model is not None and not isinstance(model, str):
        raise ValueError(f"model must be text such as 0.6b or 1.7b, got {model!r}")
    diarize = _flag(_pick(args.diarize, cfg["diarize"]), "diarize")

    # The three counts are one setting: any of them on the command line replaces all
    # of the config's, and the config's only apply when diarizing.
    cli_counts = {k: getattr(args, k) for k in _SPEAKER_KEYS}
    if any(v is not None for v in cli_counts.values()):
        if not diarize:
            raise ValueError("--num-speakers / --min-speakers / --max-speakers need --diarize")
        raw = cli_counts
    else:
        raw = {k: (cfg[k] if diarize else None) for k in _SPEAKER_KEYS}
    counts = {k: _speaker_count(raw[k], "--" + k.replace("_", "-")) for k in _SPEAKER_KEYS}

    options = Options(
        model=model or None,
        timestamps=_flag(_pick(args.timestamps, cfg["timestamps"]), "timestamps"),
        diarize=diarize,
        **counts,
    )
    if options.num_speakers is not None and (options.min_speakers or options.max_speakers):
        raise ValueError("use either --num-speakers or --min/--max-speakers, not both")
    if (options.min_speakers and options.max_speakers
            and options.min_speakers > options.max_speakers):
        raise ValueError("--min-speakers is larger than --max-speakers")
    return options, fmt


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.path is None and not args.doctor:
        print(f"{parser.format_usage().rstrip()}\n\n{EXAMPLES}\n"
              "All options: media-transcribe --help", file=sys.stderr)
        return 2
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as e:
        print(f"error: bad config: {e}", file=sys.stderr)
        return 2

    engine_name = args.engine or cfg["engine"]
    language = args.language or cfg["language"] or "zh"   # `language: null` -> default
    try:
        for name, value in (("engine", engine_name), ("language", language)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be text, got {value!r}")
        options, fmt = _build_options(args, cfg)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.doctor:
        from .doctor import run as run_doctor

        return run_doctor(engine_name, options, language=language, fmt=fmt)

    context_file = args.context_file or cfg.get("context_file")
    context_inline = args.context if args.context is not None else cfg.get("context")
    try:
        recursive = _flag(_pick(args.recursive, cfg["recursive"]), "recursive")
        for name, value in (("output_dir", cfg.get("output_dir")),
                            ("context_file", context_file), ("context", context_inline)):
            if value is not None and not isinstance(value, (str, Path)):
                raise ValueError(f"{name} must be text, got {value!r}")
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    root: Path = args.path
    if not root.exists():
        print(f"error: path not found: {root}", file=sys.stderr)
        return 2
    if root.is_file() and not is_media(root):
        print(f"error: not a supported media file: {root}", file=sys.stderr)
        return 2

    if args.out is not None:
        out_dir = args.out
    elif cfg.get("output_dir"):
        out_dir = Path(cfg["output_dir"])
    else:
        base_dir = root.parent if root.is_file() else root
        out_dir = base_dir / "transcripts"

    try:
        engine = get_engine(engine_name)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    needed = options.requested_features() | (
        {FEATURE_TIMESTAMPS} if fmt in TIMED_FORMATS else set())
    missing = needed - set(getattr(engine, "features", ()))
    if missing:
        print(f"error: engine {engine_name!r} does not support: "
              f"{', '.join(sorted(missing))}", file=sys.stderr)
        return 2

    ok, reason = engine.is_available()
    if not ok:
        if args.dry_run:
            print(f"warning: engine {engine_name!r} unavailable: {reason} "
                  "(dry-run continues)", file=sys.stderr)
        else:
            print(f"error: engine {engine_name!r} unavailable: {reason}", file=sys.stderr)
            return 2

    try:
        context_terms = _resolve_context(context_file, context_inline)
    except (OSError, UnicodeDecodeError) as e:
        print(f"error: cannot read context file {context_file}: {e}", file=sys.stderr)
        return 2

    if not args.dry_run:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"error: cannot create output folder {out_dir}: {e}", file=sys.stderr)
            return 2
        if not os.access(out_dir, os.W_OK):
            print(f"error: output folder is not writable: {out_dir}", file=sys.stderr)
            return 2

    if args.verbose:
        counts = {k: getattr(options, k) for k in ("num_speakers", "min_speakers", "max_speakers")}
        speakers = ", ".join(f"{k}={v}" for k, v in counts.items() if v is not None)
        print(f"settings: engine={engine_name} model={options.model or 'default'} "
              f"lang={language} format={fmt} diarize={'on' if options.diarize else 'off'}"
              f"{' (' + speakers + ')' if speakers else ''} "
              f"timestamps={'on' if options.timestamps else 'off'} recursive={recursive} "
              f"context_terms={len(context_terms)} out={out_dir}", file=sys.stderr)

    from .batch import run

    results, code = run(
        root,
        engine=engine,
        out_dir=out_dir,
        language=language,
        context_terms=context_terms,
        recursive=recursive,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        log=print,
        options=options,
        fmt=fmt,
    )

    done = sum(1 for r in results if r.status == "done")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    planned = sum(1 for r in results if r.status == "planned")
    if args.dry_run:
        print(f"[dry-run] {planned} to transcribe, {skipped} already done",
              file=sys.stderr)
    else:
        print(f"done={done} skipped={skipped} failed={failed} -> {out_dir}",
              file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
