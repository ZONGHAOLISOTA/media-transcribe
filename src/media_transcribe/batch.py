"""Batch discovery, output-path mapping, collision pre-flight, resumable run."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from . import media, render
from .clean import clean, clean_segments
from .engines.base import Engine, Options


@dataclass
class Item:
    src: Path     # input media file
    rel: Path     # path relative to the scan base (for tree mirroring)
    target: Path  # output path (.txt, or the --format extension)


@dataclass
class Result:
    src: str
    target: str
    status: str   # "done" | "skipped" | "failed" | "planned"
    error: str = ""


def discover(root: Path, *, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root] if media.is_media(root) else []
    globber = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in globber if p.is_file() and media.is_media(p))


def plan(root: Path, files: list[Path], out_dir: Path, suffix: str = ".txt") -> list[Item]:
    """Map each input to ``<out>/<relpath><suffix>``, mirroring the input tree."""
    base = root.parent if root.is_file() else root
    items: list[Item] = []
    for f in files:
        try:
            rel = f.relative_to(base)
        except ValueError:
            rel = Path(f.name)
        target = (out_dir / rel).with_suffix(suffix)
        items.append(Item(src=f, rel=rel, target=target))
    return items


def find_collisions(items: list[Item]) -> dict[Path, list[Path]]:
    """Return {target: [srcs]} for any target reached by more than one input."""
    seen: dict[Path, list[Path]] = {}
    for it in items:
        seen.setdefault(it.target, []).append(it.src)
    return {t: srcs for t, srcs in seen.items() if len(srcs) > 1}


def _process_one(
    it: Item, *, engine: Engine, language: str, context_terms: list[str],
    options: Options, fmt: str,
) -> str:
    with tempfile.TemporaryDirectory(prefix="media-transcribe-") as td:
        wav = Path(td) / (it.src.stem + ".wav")
        media.extract_wav(it.src, wav)
        if render.needs_segments(options, fmt):
            segments = engine.transcribe_segments(
                wav, language=language, context=context_terms, options=options
            )
            return render.render_segments(
                clean_segments(segments, context_terms),
                fmt=fmt, language=language, timestamps=options.timestamps,
            )
        # Only pass ``options`` when one was requested, so engines written for
        # v0.1 (no ``options`` parameter) keep working.
        extra = {"options": options} if options.requested_features() else {}
        raw = engine.transcribe(wav, language=language, context=context_terms, **extra)
    return render.render_text(clean(raw, context_terms), fmt=fmt, language=language)


def _write_manifest(out_dir: Path, results: list[Result]) -> None:
    mdir = out_dir / ".media-transcribe"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "run.json").write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run(
    root: Path,
    *,
    engine: Engine,
    out_dir: Path,
    language: str,
    context_terms: list[str],
    recursive: bool,
    overwrite: bool,
    dry_run: bool,
    log: Callable[[str], None] = print,
    options: Options | None = None,
    fmt: str = "txt",
) -> tuple[list[Result], int]:
    """Process ``root``. Returns (results, exit_code)."""
    options = options or Options()
    files = discover(root, recursive=recursive)
    items = plan(root, files, out_dir, suffix=f".{fmt}")

    collisions = find_collisions(items)
    if collisions:
        for target, srcs in collisions.items():
            log(f"COLLISION: {len(srcs)} inputs map to {target}")
            for s in srcs:
                log(f"  - {s}")
        if not dry_run:
            return [], 2

    results: list[Result] = []
    failures = 0
    total = len(items)
    for i, it in enumerate(items, 1):
        prefix = f"[{i}/{total}] {it.rel}"
        if it.target.exists() and it.target.stat().st_size > 0 and not overwrite:
            log(f"{prefix}  skip (exists; --overwrite to redo)")
            results.append(Result(str(it.src), str(it.target), "skipped"))
            continue
        if dry_run:
            log(f"{prefix}  would transcribe -> {it.target}")
            results.append(Result(str(it.src), str(it.target), "planned"))
            continue
        try:
            log(f"{prefix}  transcribing...")
            text = _process_one(
                it, engine=engine, language=language, context_terms=context_terms,
                options=options, fmt=fmt,
            )
            it.target.parent.mkdir(parents=True, exist_ok=True)
            # Write then rename, so an interrupted run never leaves a partial file
            # that the resume check would mistake for a finished transcript.
            part = it.target.with_name(it.target.name + ".part")
            part.write_text(text, encoding="utf-8")
            part.replace(it.target)
            results.append(Result(str(it.src), str(it.target), "done"))
        except Exception as e:  # noqa: BLE001 - record and continue the batch
            failures += 1
            log(f"{prefix}  FAILED: {e}")
            results.append(Result(str(it.src), str(it.target), "failed", str(e)))

    if not dry_run:
        _write_manifest(out_dir, results)
    return results, (1 if failures else 0)
