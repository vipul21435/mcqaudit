"""mcqaudit: what a multiple-choice benchmark gives away before any model reads the question."""

from .choices import ChoicesOnlyResult, choices_only
from .cues import CueResult, default_heuristics, evaluate, evaluate_all
from .data import Dataset, Item, LoadError, items_from_records, load
from .hygiene import HygieneReport, OverlapReport, hygiene, split_overlap
from .keyaudit import KeyAudit, audit_key, position_test, runs_test
from .shuffle import balance_positions, shuffle_options, write_jsonl

__version__ = "0.1.0"

__all__ = [
    "ChoicesOnlyResult",
    "CueResult",
    "Dataset",
    "HygieneReport",
    "Item",
    "KeyAudit",
    "LoadError",
    "OverlapReport",
    "__version__",
    "audit_key",
    "balance_positions",
    "choices_only",
    "default_heuristics",
    "evaluate",
    "evaluate_all",
    "hygiene",
    "items_from_records",
    "load",
    "position_test",
    "runs_test",
    "shuffle_options",
    "split_overlap",
    "write_jsonl",
]
