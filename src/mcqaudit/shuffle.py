"""Copies of a dataset with the options reordered and the key updated."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from .data import Item


def shuffle_options(items: Sequence[Item], seed: int = 0) -> list[Item]:
    """Each item's options in a random order (seeded); the answer index follows its text."""
    rng = random.Random(seed)
    out = []
    for item in items:
        order = list(range(item.n_options))
        rng.shuffle(order)
        options = tuple(item.options[j] for j in order)
        out.append(Item(item.id, item.question, options, order.index(item.answer), item.subject))
    return out


def balance_positions(items: Sequence[Item], seed: int = 0) -> list[Item]:
    """Move each answer so that positions are as even as possible within each option count.

    Within a group of items with ``k`` options the target positions are
    ``0, 1, ..., k-1`` repeated and shuffled (seeded), so counts differ by at most
    one. The correct option is taken out and reinserted at its target; the
    distractors keep their relative order.
    """
    rng = random.Random(seed)
    groups: dict[int, list[int]] = defaultdict(list)
    for i, item in enumerate(items):
        groups[item.n_options].append(i)
    targets: dict[int, int] = {}
    for k in sorted(groups):
        members = groups[k]
        slots = [j % k for j in range(len(members))]
        rng.shuffle(slots)
        targets.update(zip(members, slots, strict=True))
    out = []
    for i, item in enumerate(items):
        rest = [o for j, o in enumerate(item.options) if j != item.answer]
        t = targets[i]
        options = tuple(rest[:t] + [item.answer_text] + rest[t:])
        out.append(Item(item.id, item.question, options, t, item.subject))
    return out


def write_jsonl(items: Sequence[Item], path: str | Path) -> None:
    """Write items as JSONL: id, question, choices, 0-based answer and subject."""
    with Path(path).open("w", encoding="utf-8") as fh:
        for item in items:
            row = {
                "id": item.id,
                "question": item.question,
                "choices": list(item.options),
                "answer": item.answer,
            }
            if item.subject is not None:
                row["subject"] = item.subject
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
