"""Command-line interface for mcqaudit."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .choices import choices_only
from .cues import evaluate_all
from .data import Dataset, LoadError, load
from .hygiene import hygiene, split_overlap
from .keyaudit import DEFAULT_EXACT_LIMIT, audit_key
from .report import (
    audit_dict,
    audit_markdown,
    choices_dict,
    choices_markdown,
    cues_dict,
    cues_markdown,
    hygiene_markdown,
    to_json,
)
from .shuffle import balance_positions, shuffle_options, write_jsonl


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

    co = sub.add_parser("choices-only", help="cross-validated classifier that sees only options")
    _common(co)
    co.add_argument("--folds", type=int, default=5, help="number of folds (default 5)")
    co.add_argument("--seed", type=int, default=0, help="seed for fold assignment")
    co.add_argument("--C", type=float, default=1.0, help="inverse L2 strength, as in sklearn")
    co.add_argument(
        "--features", type=int, default=2**12, help="hashed columns per position (default 4096)"
    )
    co.add_argument(
        "--max-iter", type=int, default=300, help="most L-BFGS iterations per fold (default 300)"
    )
    co.add_argument(
        "--group",
        choices=("subject", "none"),
        default="subject",
        help="group folds by subject when the data has subjects (default)",
    )
    co.add_argument("--confidence", type=float, default=0.95, help="interval level")

    hy = sub.add_parser("hygiene", help="duplicate, empty and leaking options; split overlap")
    _common(hy)
    hy.add_argument("--against", help="a second file (another split) to check for overlap")
    hy.add_argument(
        "--threshold", type=float, default=0.8, help="Jaccard threshold for near duplicates"
    )

    sh = sub.add_parser("shuffle", help="write a copy with options reordered and the key updated")
    _common(sh)
    sh.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    sh.add_argument(
        "--balanced",
        action="store_true",
        help="place answers so positions are as even as possible instead of shuffling",
    )
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
    elif args.command == "choices-only":
        res = choices_only(
            ds.items,
            n_folds=args.folds,
            seed=args.seed,
            C=args.C,
            n_features=args.features,
            group=args.group,
            confidence=args.confidence,
            max_iter=args.max_iter,
        )
        if args.format == "json":
            text = to_json({"source": ds.source, **choices_dict(res)})
        else:
            text = choices_markdown(res, Path(ds.source).name, args.confidence)
    elif args.command == "hygiene":
        rep = hygiene(ds.items)
        overlap = None
        if args.against:
            try:
                other = load(args.against, fmt=args.input_format, answer_base=args.answer_base)
            except (LoadError, OSError) as exc:
                print(f"mcqaudit: {exc}", file=sys.stderr)
                return 2
            overlap = split_overlap(other.items, ds.items, threshold=args.threshold)
        if args.format == "json":
            data = {"source": ds.source, **asdict(rep)}
            if overlap is not None:
                data["overlap"] = {"against": other.source, **asdict(overlap)}
            text = to_json(data)
        else:
            text = hygiene_markdown(rep, overlap, Path(ds.source).name)
    elif args.command == "shuffle":
        if not args.output:
            print("mcqaudit: shuffle needs -o/--output for the new file", file=sys.stderr)
            return 2
        if args.balanced:
            new = balance_positions(ds.items, args.seed)
        else:
            new = shuffle_options(ds.items, args.seed)
        write_jsonl(new, args.output)
        before = audit_key(ds.items, by_subject=False).positions
        after = audit_key(new, by_subject=False).positions
        print(
            f"wrote {len(new)} items to {args.output}; position test p {before.p_value:.3g} "
            f"before, {after.p_value:.3g} after",
            file=sys.stderr,
        )
        return 0
    else:  # pragma: no cover - argparse enforces the choices
        parser.error(f"unknown command {args.command}")
    _emit(text, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
