"""Command-line interface for mcqaudit."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .cues import evaluate_all
from .data import Dataset, LoadError, load
from .keyaudit import DEFAULT_EXACT_LIMIT, audit_key
from .report import audit_dict, audit_markdown, cues_dict, cues_markdown, to_json


def _answer_base(value: str) -> str | int:
    if value == "auto":
        return "auto"
    if value in ("0", "1"):
        return int(value)
    raise argparse.ArgumentTypeError("answer base must be auto, 0 or 1")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="dataset file (.jsonl, .json, .csv, .tsv or .parquet)")
    parser.add_argument("--input-format", help="override the format given by the extension")
    parser.add_argument(
        "--answer-base",
        type=_answer_base,
        default="auto",
        help="whether integer answers count from 0 or 1 (default: auto)",
    )
    parser.add_argument("--format", choices=("md", "json"), default="md", help="output format")
    parser.add_argument("-o", "--output", help="write the report here instead of stdout")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcqaudit",
        description="Audit a multiple-choice dataset for cues that need no model to exploit.",
    )
    parser.add_argument("--version", action="version", version=f"mcqaudit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="answer-position balance and key order")
    _common(audit)
    audit.add_argument(
        "--method",
        choices=("auto", "exact", "chi-square"),
        default="auto",
        help="position test; auto is exact when the tables can be enumerated",
    )
    audit.add_argument(
        "--exact-limit",
        type=int,
        default=DEFAULT_EXACT_LIMIT,
        help=f"most tables to enumerate for the exact test (default {DEFAULT_EXACT_LIMIT})",
    )
    audit.add_argument("--no-subjects", action="store_true", help="skip the per-subject tables")

    cues = sub.add_parser("cues", help="accuracy of question-free heuristics against chance")
    _common(cues)
    cues.add_argument("--confidence", type=float, default=0.95, help="interval level")
    return parser


def _load(args: argparse.Namespace) -> Dataset:
    return load(args.path, fmt=args.input_format, answer_base=args.answer_base)


def _emit(text: str, output: str | None) -> None:
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        ds = _load(args)
    except (LoadError, OSError) as exc:
        print(f"mcqaudit: {exc}", file=sys.stderr)
        return 2
    for note in ds.notes:
        print(f"mcqaudit: {note}", file=sys.stderr)

    if args.command == "audit":
        result = audit_key(
            ds.items,
            by_subject=not args.no_subjects,
            method=args.method,
            exact_limit=args.exact_limit,
        )
        if args.format == "json":
            text = to_json({"source": ds.source, **audit_dict(result)})
        else:
            text = audit_markdown(result, Path(ds.source).name)
    elif args.command == "cues":
        results = evaluate_all(ds.items, confidence=args.confidence)
        if args.format == "json":
            text = to_json({"source": ds.source, "cues": cues_dict(results)})
        else:
            text = cues_markdown(results, Path(ds.source).name, args.confidence)
    else:  # pragma: no cover - argparse enforces the choices
        parser.error(f"unknown command {args.command}")
    _emit(text, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
