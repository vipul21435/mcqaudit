"""Question-free heuristics and how far they score above chance.

A heuristic looks at one item and names the options it would pick. When it
names several (a tie), it is scored by its expected credit under random
tie-breaking, ``1/|C|`` when the answer is in the set ``C``; when it names every
option, or none, it abstains on that item. Accuracy is reported over the items
it covers, with a Wilson interval.

Against chance: if the key position were independent of what the heuristic
looks at, an item with ``k`` options would give credit with mean ``1/k`` and
variance ``1/(k|C|) - 1/k^2``. The z statistic sums both over covered items and
the one-sided p-value uses the normal approximation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .data import LETTERS, Item
from .stats import normal_upper_p, wilson_interval

Heuristic = Callable[[Item], frozenset[int]]


def _extremes(values: Sequence[float], pick_max: bool) -> frozenset[int]:
    target = max(values) if pick_max else min(values)
    return frozenset(i for i, v in enumerate(values) if v == target)


def _length(option: str) -> int:
    return len(option.strip())


def longest(item: Item) -> frozenset[int]:
    """The option with the most characters."""
    return _extremes([_length(o) for o in item.options], pick_max=True)


def shortest(item: Item) -> frozenset[int]:
    """The option with the fewest characters."""
    return _extremes([_length(o) for o in item.options], pick_max=False)


def position(k: int) -> Heuristic:
    """Always the option at 0-based position ``k``; abstains when an item has fewer options."""

    def pick(item: Item) -> frozenset[int]:
        return frozenset({k}) if k < item.n_options else frozenset()

    pick.__name__ = f"position_{LETTERS[k]}"
    pick.__doc__ = f"Always option {LETTERS[k]}."
    return pick


def default_heuristics(max_options: int) -> dict[str, Heuristic]:
    """The fixed set of heuristics, named as they appear in reports."""
    table: dict[str, Heuristic] = {
        f"always {LETTERS[k]}": position(k) for k in range(max(max_options, 2))
    }
    table["longest option"] = longest
    table["shortest option"] = shortest
    return table


@dataclass
class CueResult:
    name: str
    n_items: int
    covered: int
    credit: float
    accuracy: float
    ci_low: float
    ci_high: float
    chance: float
    z: float | None
    p_value: float | None

    @property
    def lift(self) -> float:
        """Accuracy minus chance, in proportion units."""
        return self.accuracy - self.chance


def evaluate(
    items: Sequence[Item], name: str, heuristic: Heuristic, *, confidence: float = 0.95
) -> CueResult:
    """Score one heuristic on ``items``."""
    covered = 0
    credit = 0.0
    chance_sum = 0.0
    var_sum = 0.0
    for item in items:
        picks = heuristic(item)
        k = item.n_options
        if not picks or len(picks) >= k:
            continue
        covered += 1
        size = len(picks)
        if item.answer in picks:
            credit += 1.0 / size
        chance_sum += 1.0 / k
        var_sum += 1.0 / (k * size) - 1.0 / (k * k)
    if covered == 0:
        return CueResult(name, len(items), 0, 0.0, 0.0, 0.0, 1.0, 0.0, None, None)
    accuracy = credit / covered
    lo, hi = wilson_interval(credit, covered, confidence)
    z = p = None
    if var_sum > 0:
        z = (credit - chance_sum) / math.sqrt(var_sum)
        p = normal_upper_p(z)
    return CueResult(
        name, len(items), covered, credit, accuracy, lo, hi, chance_sum / covered, z, p
    )


def evaluate_all(
    items: Sequence[Item],
    heuristics: dict[str, Heuristic] | None = None,
    *,
    confidence: float = 0.95,
) -> list[CueResult]:
    """Score every heuristic, best lift first."""
    if heuristics is None:
        heuristics = default_heuristics(max((i.n_options for i in items), default=4))
    results = [evaluate(items, n, h, confidence=confidence) for n, h in heuristics.items()]
    results.sort(key=lambda r: (r.covered == 0, -r.lift))
    return results
