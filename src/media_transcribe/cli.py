"""Command-line entry point for media-transcribe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import load_config
from .engines.base import available_engines, get_engine
from .media import is_media


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
        description="AI-native, engine-pluggable media-to-text.",
    )
    p.add_argument("path", type=Path, help="file or directory (video or audio)")
    p.add_argument("--engine", default=None,
                   help=f"ASR engine (available: {', '.join(available_engines()) or 'none'})")
    p.add_argument("--lang", dest="language", default=None, help="language (default: zh)")
    p.add_argument("--context-file", dest="context_file", default=None,
                   help="domain-vocab file (whitespace/newline-separated terms)")
    p.add_argument("--context", dest="context", default=None,
                   help="inline domain-vocab terms (merged with --context-file)")
    p.add_argument("--out", dest="out", type=Path, default=None, help="output directory")
    p.add_argument("--recursive", action="store_true", default=None,
                   help="recurse into subdirectories")
    p.add_argument("--overwrite", action="store_true",
                   help="re-transcribe even if a transcript exists")
    p.add_argument("--dry-run", action="store_true", help="list planned work, do nothing")
    p.add_argument("--config", type=Path, default=None, help="YAML config file")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"media-transcribe {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as e:
        print(f"error: bad config: {e}", file=sys.stderr)
        return 2

    engine_name = args.engine or cfg["engine"]
    language = args.language or cfg["language"]
    recursive = args.recursive if args.recursive is not None else bool(cfg["recursive"])
    context_file = args.context_file or cfg.get("context_file")
    context_inline = args.context if args.context is not None else cfg.get("context")

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
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

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
