"""Read multiple-choice items from JSONL, JSON, CSV or Parquet files.

Each record is mapped to an :class:`Item` by the first layout that fits it:

``list``
    ``question`` plus ``choices`` (or ``options``) as a list of strings and
    ``answer`` as an index or a letter (the MMLU layout on the Hugging Face hub).
``text-label``
    ``choices`` as ``{"text": [...], "label": [...]}`` and an ``answerKey``
    naming one of the labels (the ARC and CommonsenseQA layout on the hub).
``stem``
    ``question`` as ``{"stem": ..., "choices": [{"text": ..., "label": ...}]}``
    with an ``answerKey`` (the original ARC and CommonsenseQA JSONL files).
``dict``
    ``options`` as a mapping from label to text, and the answer as a label,
    an index or the text itself (``answer_idx`` or ``answer``).
``columns``
    one column per option, named ``A``, ``B``, ... or ``opa``, ``opb``, ... or
    ``option_a``, ``option_1``, ...; the usual layout of CSV files.

Integer answers are 0-based unless ``answer_base`` says otherwise. With
``answer_base="auto"`` a file whose integer answers never use 0 and at least
once equal the option count is read as 1-based.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

QUESTION_KEYS = ("question", "stem", "prompt", "query")
ANSWER_KEYS = ("answerKey", "answer_idx", "answer", "label", "target", "cop", "correct")
SUBJECT_KEYS = ("subject", "subject_name", "category", "topic", "domain")
ID_KEYS = ("id", "qid", "item_id", "question_id")

_OPTION_COLUMN = re.compile(r"^(?:option|op|choice)[ _-]?([a-z]|\d{1,2})$", re.IGNORECASE)

AnswerBase = Literal["auto", 0, 1]


class LoadError(ValueError):
    """Raised when a record cannot be read as a multiple-choice item."""


@dataclass(frozen=True)
class Item:
    """One multiple-choice item.

    ``answer`` is the 0-based index of the correct option in ``options``.
    """

    id: str
    question: str
    options: tuple[str, ...]
    answer: int
    subject: str | None = None

    @property
    def n_options(self) -> int:
        return len(self.options)

    @property
    def answer_text(self) -> str:
        return self.options[self.answer]

    @property
    def answer_letter(self) -> str:
        return LETTERS[self.answer]


@dataclass
class Dataset:
    """Items read from one file, with a note of how they were read."""

    items: list[Item]
    source: str
    layouts: dict[str, int] = field(default_factory=dict)
    answer_base: int = 0
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[Item]:
        return iter(self.items)


@dataclass
class _Raw:
    index: int
    id: str
    question: str
    options: list[str]
    labels: list[str] | None
    answer: Any
    subject: str | None
    layout: str


def _first(record: Mapping[str, Any], keys: Sequence[str]) -> tuple[str | None, Any]:
    for key in keys:
        if key in record and record[key] is not None and record[key] != "":
            return key, record[key]
    return None, None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN from pandas-written files
        return ""
    return str(value)


def _maybe_json_list(value: Any) -> Any:
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, list):
            return parsed
    return value


def _column_options(record: Mapping[str, Any]) -> tuple[list[str], list[str]] | None:
    letters = []
    for letter in LETTERS:
        if letter in record:
            letters.append(letter)
        else:
            break
    if len(letters) >= 2:
        return [_text(record[k]) for k in letters], letters

    found: list[tuple[int, str, str]] = []
    for key in record:
        match = _OPTION_COLUMN.match(str(key))
        if not match:
            continue
        tag = match.group(1)
        order = int(tag) if tag.isdigit() else LETTERS.index(tag.upper())
        found.append((order, str(key), tag.upper()))
    if len(found) >= 2:
        found.sort()
        return [_text(record[k]) for _, k, _ in found], [t for _, _, t in found]
    return None


def _parse_record(record: Mapping[str, Any], index: int) -> _Raw:
    if not isinstance(record, Mapping):
        raise LoadError(f"record {index + 1}: expected an object, got {type(record).__name__}")

    _, raw_id = _first(record, ID_KEYS)
    item_id = _text(raw_id) if raw_id is not None else f"item-{index + 1}"
    _, subject = _first(record, SUBJECT_KEYS)
    subject = _text(subject) if subject is not None else None
    _, answer = _first(record, ANSWER_KEYS)

    question_value = record.get("question")
    if isinstance(question_value, Mapping) and "choices" in question_value:
        choices = question_value["choices"]
        if not isinstance(choices, list) or not all(isinstance(c, Mapping) for c in choices):
            raise LoadError(f"record {index + 1}: question.choices must be a list of objects")
        return _Raw(
            index,
            item_id,
            _text(question_value.get("stem", "")),
            [_text(c.get("text")) for c in choices],
            [_text(c.get("label")) for c in choices],
            answer,
            subject,
            "stem",
        )

    _, question = _first(record, QUESTION_KEYS)
    question = _text(question)

    for key in ("choices", "options", "endings"):
        if key not in record:
            continue
        value = _maybe_json_list(record[key])
        if isinstance(value, Mapping) and "text" in value:
            texts = list(value["text"])
            labels = list(value.get("label") or LETTERS[: len(texts)])
            if len(labels) != len(texts):
                raise LoadError(f"record {index + 1}: {key}.text and {key}.label differ in length")
            return _Raw(
                index,
                item_id,
                question,
                [_text(t) for t in texts],
                [_text(lab) for lab in labels],
                answer,
                subject,
                "text-label",
            )
        if isinstance(value, Mapping):
            return _Raw(
                index,
                item_id,
                question,
                [_text(v) for v in value.values()],
                [_text(k) for k in value],
                answer,
                subject,
                "dict",
            )
        if isinstance(value, (list, tuple)):
            return _Raw(
                index, item_id, question, [_text(v) for v in value], None, answer, subject, "list"
            )

    columns = _column_options(record)
    if columns is not None:
        options, labels = columns
        return _Raw(index, item_id, question, options, labels, answer, subject, "columns")

    raise LoadError(f"record {index + 1}: no options found (looked for choices, options, A/B/...)")


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
        return int(value)
    return None


def _resolve_answer(raw: _Raw, base: int) -> int:
    answer = raw.answer
    if answer is None:
        raise LoadError(f"record {raw.index + 1}: no answer field")
    n = len(raw.options)

    if raw.labels is not None and isinstance(answer, str):
        stripped = answer.strip()
        if stripped in raw.labels:
            return raw.labels.index(stripped)
        upper = [lab.upper() for lab in raw.labels]
        if stripped.upper() in upper:
            return upper.index(stripped.upper())

    as_int = _as_int(answer)
    if as_int is not None:
        idx = as_int - base
    elif isinstance(answer, str) and len(answer.strip()) == 1 and answer.strip().isalpha():
        idx = LETTERS.index(answer.strip().upper())
    elif isinstance(answer, str) and answer in raw.options:
        idx = raw.options.index(answer)
    else:
        raise LoadError(f"record {raw.index + 1}: cannot read answer {answer!r}")

    if not 0 <= idx < n:
        raise LoadError(
            f"record {raw.index + 1}: answer {answer!r} is out of range for {n} options"
        )
    return idx


def _detect_base(raws: Sequence[_Raw]) -> tuple[int, str | None]:
    ints = []
    for raw in raws:
        is_label = isinstance(raw.answer, str) and raw.answer.strip() in (raw.labels or ())
        if is_label:
            continue
        value = _as_int(raw.answer)
        if value is not None:
            ints.append((value, len(raw.options)))
    if not ints:
        return 0, None
    if any(v == 0 for v, _ in ints):
        return 0, None
    if any(v == n for v, n in ints):
        return 1, "integer answers never use 0 and reach the option count: read as 1-based"
    return 0, "integer answers never use 0 but never reach the option count: read as 0-based"


def items_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    source: str = "<records>",
    answer_base: AnswerBase = "auto",
) -> Dataset:
    """Build a :class:`Dataset` from already-parsed records."""
    raws = [_parse_record(rec, i) for i, rec in enumerate(records)]
    notes: list[str] = []
    if answer_base == "auto":
        base, note = _detect_base(raws)
        if note:
            notes.append(note)
    elif answer_base in (0, 1):
        base = int(answer_base)
    else:
        raise ValueError(f"answer_base must be 'auto', 0 or 1, not {answer_base!r}")

    items = []
    layouts: dict[str, int] = {}
    for raw in raws:
        if len(raw.options) < 2:
            raise LoadError(f"record {raw.index + 1}: fewer than two options")
        idx = _resolve_answer(raw, base)
        items.append(Item(raw.id, raw.question, tuple(raw.options), idx, raw.subject))
        layouts[raw.layout] = layouts.get(raw.layout, 0) + 1
    return Dataset(items, source, layouts, base, notes)


def read_records(path: str | Path, fmt: str | None = None) -> list[dict[str, Any]]:
    """Read raw records from a JSONL, JSON, CSV/TSV or Parquet file."""
    path = Path(path)
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    if fmt in ("jsonl", "ndjson"):
        records = []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise LoadError(f"{path}:{lineno}: invalid JSON ({exc.msg})") from exc
        return records
    if fmt == "json":
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, Mapping):
            for key in ("data", "items", "rows", "examples"):
                if isinstance(data.get(key), list):
                    return list(data[key])
            raise LoadError(f"{path}: JSON object without a list under data/items/rows/examples")
        if not isinstance(data, list):
            raise LoadError(f"{path}: expected a JSON list of records")
        return data
    if fmt in ("csv", "tsv"):
        delimiter = "\t" if fmt == "tsv" else ","
        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh, delimiter=delimiter))
    if fmt == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - depends on the environment
            msg = "reading Parquet needs pyarrow: pip install 'mcqaudit[parquet]'"
            raise LoadError(msg) from exc
        return pq.read_table(path).to_pylist()
    raise LoadError(f"{path}: unknown format {fmt!r} (use jsonl, json, csv, tsv or parquet)")


def load(path: str | Path, *, fmt: str | None = None, answer_base: AnswerBase = "auto") -> Dataset:
    """Read a multiple-choice dataset from ``path``; the format follows the extension."""
    return items_from_records(read_records(path, fmt), source=str(path), answer_base=answer_base)
