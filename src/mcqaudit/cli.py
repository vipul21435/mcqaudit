"""Command-line interface for mcqaudit."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcqaudit",
        description="Audit a multiple-choice dataset for cues that need no model to exploit.",
    )
    parser.add_argument("--version", action="version", version=f"mcqaudit {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
